"""unite2_pro_triviaqa.py — TriviaQA 专用 pro 版 ensemble decoding。

相对原 unite2_pro.py 的修复点（与 unite2_triviaqa.py 共享的修复 + pro 特有修复）：
  共享：
    1. 早停条件从 "<END>" 死代码改为可触发的 \\n\\n / Question: / "Final answer: <X>\\n" 检测。
    2. collate_fn 拼接改用单换行；模板结尾改为 Explanation:，引导模型先短解释再给答案。
    3. max_new_tokens 默认 64（解释+答案两段，需要更长步数）。
    4. 仅 TriviaQA，移除其他数据集路由。
    5. 答案抽取：先匹配 "Final answer: X" 行作为 pred。
  pro 特有：
    6. 动态权重的 HISTORY_WINDOW 默认 5（TriviaQA 答案短，窗口 20 太慢热）。
    7. 冷启动修复：history < HISTORY_WINDOW 时不裁剪到 [MIN_WEIGHT, 1-MIN_WEIGHT]，
       history == 0 时仍走 0.5/0.5 等权起步；按已观察分数即时计算。
    8. 冗余融合跳过：top-1 token 一致时跳过融合（pro 原版就有，保留）。
"""

from tqdm import tqdm
import re
import json

import torch
import argparse

from utils.ans_process import qa_parse_pred_ans
from accelerate import Accelerator
from torch.utils.data import DataLoader
from accelerate.utils import gather_object

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset


PROMPT_TEMPLATE = "\nQuestion: {question}\nExplanation:"

# 早停触发：模型在生成完答案后开始"接着写"。
# 不能用 "Explanation:" 当 stop pattern——它是 prompt 的结尾。
# "Final answer:" 也不当 stop pattern——模型要写完它才算完。
# 真正能用的信号：\n\n 出现 / Question: 出现 / Final answer 那一行已写完。
STOP_STRINGS = ("\n\n", "Question:")
FINAL_ANSWER_LINE_RE = re.compile(r'final answer:\s*([^\n]+)', re.IGNORECASE)


def triviaqa_collate_fn(batch):
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        prompt_q = prompt_complex + PROMPT_TEMPLATE.format(question=ques)
        questions.append(prompt_q)
        answers.append(b["answer"])
    return questions, answers


# ---------------------------------------------------------------------------
# 词汇层工具（与 unite2_triviaqa.py 共享）
# ---------------------------------------------------------------------------
def get_top_k_tokens(outputs, tokenizer, k=10):
    logits = outputs.logits[0]
    top_k_indices = torch.topk(logits, k).indices
    probs = logits.tolist()

    top_k_probs = []
    for idx, prob in zip(top_k_indices, probs):
        top_k_probs.append([prob[i] for i in idx])

    top_k_tokens = []
    for indices in top_k_indices:
        top_k_tokens.append(
            [tokenizer.convert_ids_to_tokens(idx.item(), skip_special_tokens=True) for idx in indices]
        )

    v1 = []
    for token, prob, id in zip(top_k_tokens, top_k_probs, top_k_indices):
        v1.append({t.replace('▁', 'Ġ').replace('<0x0A>', '/n').replace('Ċ', '/n'): [p, int(i)]
                   for t, p, i in zip(token, prob, id)})
    return v1


def get_union_vocab(v1, v2):
    return [list(set(v1_tokens.keys()) | set(v2_tokens.keys()))
            for v1_tokens, v2_tokens in zip(v1, v2)]


def update_vocab(v1, vu, tokenizer, logits, model_name):
    for vu_token, v1_token, logit_ele in zip(vu, v1, logits):
        v1_token_ids = [item[1] for item in v1_token.values()]
        for token in vu_token:
            if token in v1_token.keys():
                continue
            if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                token = token.replace('Ġ', '▁')
            if token != '':
                subtoken_id = tokenizer.convert_tokens_to_ids(token)
                if subtoken_id != 0 and subtoken_id is not None:
                    logit = logit_ele[subtoken_id]
                else:
                    subtokens = tokenizer.tokenize(token)
                    subtoken_id = tokenizer.convert_tokens_to_ids(subtokens)[0]
                    logit = logit_ele[subtoken_id]
            else:
                blank_id = {'llama3': 220, 'qwen2': 220, 'llama2': 29871,
                            'mistral': 29473, 'deepseek': 207, 'openchat': 28705,
                            'glm': 128}.get(model_name.split('_')[0] if '_' in model_name else model_name, 220)
                logit = logit_ele[blank_id]
                subtoken_id = blank_id

            if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                v1_token[token.replace('▁', 'Ġ')] = [logit, subtoken_id]
            else:
                if subtoken_id not in v1_token_ids:
                    v1_token[token] = [logit, subtoken_id]
                    v1_token_ids.append(subtoken_id)
                else:
                    v1_token[token] = [0, subtoken_id]
    return vocab_softmax(v1)


def vocab_softmax(v1):
    v1_new = []
    for element in v1:
        ele = {}
        values = list(element.values())
        vals0 = [v[0] for v in values]
        vals1 = [v[1] for v in values]
        vals0 = torch.softmax(torch.tensor(vals0), dim=0)
        for token, prob, ids in zip(element.keys(), vals0, vals1):
            ele[token] = [prob, ids]
        v1_new.append(ele)
    return v1_new


def average_and_sample(v1, v2, lamda, tokenizer):
    next_token, v_avg, next_token_id1, next_token_id2 = [], [], [], []
    for element_v1, element_v2 in zip(v1, v2):
        assert len(element_v1) == len(element_v2)
        v_new = {t: [lamda * element_v1[t][0] + (1 - lamda) * element_v2[t][0],
                     element_v1[t][1]] for t in element_v1}
        v_avg.append(v_new)
        probs = [v[0] for v in v_new.values()]
        sample_index = probs.index(max(probs))
        for i, item1 in enumerate(v_new.keys()):
            if i == sample_index:
                next_token.append(tokenizer.convert_ids_to_tokens(element_v1[item1][1]))
                next_token_id1.append(element_v1[item1][1])
                next_token_id2.append(element_v2[item1][1])
    return next_token, v_avg, next_token_id1, next_token_id2


# ---------------------------------------------------------------------------
# 早停 + 答案抽取
# ---------------------------------------------------------------------------
def _should_stop(decoded_text: str) -> bool:
    if re.search(r'final answer:\s*[^\n]+\n', decoded_text, re.IGNORECASE):
        return True
    for pat in STOP_STRINGS:
        if pat in decoded_text:
            return True
    return False


def _truncate_output(text: str) -> str:
    # 先抽 Final answer: X 行再做空白归一化——
    # 否则 \n 会被 normalize 成空格，[^\n]+ 失去终止符，会越界匹配到下一题。
    m = FINAL_ANSWER_LINE_RE.search(text)
    if m:
        return ' '.join(m.group(1).split()).strip()
    text = ' '.join(text.split())
    cut_points = []
    for sep in ['\n\nQuestion:', '\nQuestion:', 'Question:']:
        idx = text.find(sep)
        if idx >= 0:
            cut_points.append(idx)
    if cut_points:
        text = text[:min(cut_points)].strip()
    return text


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def ensemble_decoding():
    fw = open(args.output_file, "w", encoding="utf-8")
    accelerator.wait_for_everyone()

    solution_list, pred_list, label_list, ori_ans_list, question_list = [], [], [], [], []
    iter_item = tqdm(ds_loader) if accelerator.is_main_process else ds_loader

    HISTORY_WINDOW = args.history_window
    MIN_WEIGHT = args.min_weight
    model1_score_history, model2_score_history = [], []

    for questions, answers in iter_item:
        output_ans = []
        inputs1 = tokenizer1(questions, padding=True, return_tensors="pt").to(device1)
        inputs2 = tokenizer2(questions, padding=True, return_tensors="pt").to(device2)
        input_ids1 = inputs1['input_ids'].to(device1)
        input_ids2 = inputs2['input_ids'].to(device2)
        attention_mask1 = inputs1['attention_mask'].to(device1)
        attention_mask2 = inputs2['attention_mask'].to(device2)
        input_length = [len(qs) for qs in input_ids1]

        past_key_values1 = None
        for i in range(args.max_new_tokens):
            if i == 0:
                outputs1 = model1.generate(input_ids=input_ids1, attention_mask=attention_mask1,
                                           generation_config=generation_config1)
                outputs2 = model2.generate(input_ids=input_ids2, attention_mask=attention_mask2,
                                           generation_config=generation_config2)
            else:
                outputs1 = model1.generate(input_ids=input_ids1, attention_mask=attention_mask1,
                                           past_key_values=past_key_values1,
                                           generation_config=generation_config1)
                outputs2 = model2.generate(input_ids=input_ids2, attention_mask=attention_mask2,
                                           generation_config=generation_config2)
            past_key_values1 = outputs1.past_key_values

            v1 = get_top_k_tokens(outputs1, tokenizer1, 10)
            v2 = get_top_k_tokens(outputs2, tokenizer2, 10)
            v1_sfmx = vocab_softmax(v1)
            v2_sfmx = vocab_softmax(v2)
            vu = get_union_vocab(v1, v2)
            v1_new = update_vocab(v1, vu, tokenizer1, outputs1.logits[0], 'qwen2')
            v2_new = update_vocab(v2, vu, tokenizer2, outputs2.logits[0], 'llama3')

            # ---- 冗余融合跳过：两个 top-1 一致时直接采纳 ----
            skip_fusion = False
            v1_dict = v1_new[0]
            v2_dict = v2_new[0]
            if v1_dict and v2_dict:
                v1_sorted = sorted(v1_dict.items(), key=lambda x: x[1][0], reverse=True)
                v2_sorted = sorted(v2_dict.items(), key=lambda x: x[1][0], reverse=True)
                if v1_sorted and v2_sorted and v1_sorted[0][0] == v2_sorted[0][0]:
                    skip_fusion = True

            # ---- 动态权重 ----
            # 修复：原 pro 版第一步强制 0.5/0.5，冷启动慢。
            # 这里即使 history=1，也按观察到的分数即时算权重（仅做概率比，无上界截断）。
            if len(model1_score_history) > 0:
                avg_score1 = sum(model1_score_history) / len(model1_score_history)
                avg_score2 = sum(model2_score_history) / len(model2_score_history)
                total_score = avg_score1 + avg_score2 + 1e-8
                w1 = avg_score1 / total_score
                # 仅在 history 满一个窗口后才收紧到 [MIN_WEIGHT, 1-MIN_WEIGHT]，
                # 避免冷启动阶段两模型还没"分出胜负"时强制偏向一边。
                if len(model1_score_history) >= HISTORY_WINDOW:
                    w1 = max(MIN_WEIGHT, min(1 - MIN_WEIGHT, w1))
                w2 = 1 - w1
            else:
                w1, w2 = 0.5, 0.5

            if skip_fusion:
                top_token = sorted(v1_dict.items(), key=lambda x: x[1][0], reverse=True)[0][0]
                next_token = [tokenizer1.convert_ids_to_tokens(v1_dict[top_token][1])]
                next_token_id1 = [v1_dict[top_token][1]]
                next_token_id2 = [v2_dict[top_token][1]]
                v_avg_dict = {t: [w1 * v1_dict[t][0] + w2 * v2_dict[t][0], v1_dict[t][1]]
                              for t in v1_dict}
                v_avg = [v_avg_dict]
            else:
                next_token, v_avg, next_token_id1, next_token_id2 = average_and_sample(
                    v1_new, v2_new, w1, tokenizer1)

            # ---- 更新历史分数 ----
            if v_avg and next_token_id1:
                final_token_str = next_token[0] if next_token else None
                if final_token_str and final_token_str in v1_new[0] and final_token_str in v2_new[0]:
                    model1_score_history.append(v1_new[0][final_token_str][0])
                    model2_score_history.append(v2_new[0][final_token_str][0])
                    model1_score_history = model1_score_history[-HISTORY_WINDOW:]
                    model2_score_history = model2_score_history[-HISTORY_WINDOW:]

            # ---- EOS ----
            if next_token_id1:
                eos_ids = tokenizer1.eos_token_id
                if not isinstance(eos_ids, list):
                    eos_ids = [eos_ids]
                if next_token_id1[0] in eos_ids:
                    break

            # ---- 推进 ----
            i1, m1 = [], []
            for pred_id, ids, mask in zip(next_token_id1, input_ids1, attention_mask1):
                i1.append(ids.tolist() + [pred_id])
                m1.append(mask.tolist() + [1])
            input_ids1 = torch.tensor(i1).to(device1)
            attention_mask1 = torch.tensor(m1).to(device1)

            generated_text = tokenizer1.decode(input_ids1[0][input_length[0]:], skip_special_tokens=True)
            if _should_stop(generated_text):
                break

            iter_input2 = tokenizer2(tokenizer1.batch_decode(input_ids1), padding=True,
                                      return_tensors="pt").to(device2)
            input_ids2 = iter_input2['input_ids'].to(device2)
            attention_mask2 = iter_input2['attention_mask'].to(device2)

        # ---- 解码 + 清理：full_text 留底，pred 抽取 ----
        # 之前 bug：output 立刻被 _truncate_output 覆盖，完整生成文本（含 Explanation）
        # 全部丢失，jsonl 里只存"John Ford"这种抽取后的结果，调试时看不到模型推理过程。
        # 现在分开存：
        #   output_ans (per-batch) = 完整生成文本（含 Explanation + Final answer 行）
        #   pred_list  (global)    = 抽取后的最终答案（送进 qa_parse_pred_ans）
        #   solution_list (global) = 完整文本（jsonl 的 pred_solution 字段）
        batch_pred_start = len(pred_list)
        for qs_len, ans in zip(input_length, input_ids1):
            full_text = tokenizer1.decode(ans[qs_len:], skip_special_tokens=True)
            pred = _truncate_output(full_text)
            output_ans.append(full_text)
            pred_list.append(pred)
        solution_list.extend(output_ans)

        label_list.extend(answers)
        ori_ans_list.extend(answers)
        # 打印：本 batch 的（完整文本前 200 字符 + 抽出的 pred），调试用
        for full, pred in zip(output_ans, pred_list[batch_pred_start:]):
            raw = full[:200].replace('\n', ' ⏎ ')
            print('---raw output---\n' + raw + ('...' if len(full) > 200 else ''))
            print('---extracted pred---:', pred)
            print('==========output========\n', pred)
        question_list.extend(questions)

    accelerator.print("======= waiting for everyone ==========")
    accelerator.wait_for_everyone()
    accelerator.print("======= start gather ==========")
    gather_pred = gather_object(pred_list)
    gather_label = gather_object(label_list)
    gather_solution = gather_object(solution_list)
    gather_ori_solution = gather_object(ori_ans_list)
    gather_qs = gather_object(question_list)

    for qs, pred, label, solution, ori_ans in zip(gather_qs, gather_pred, gather_label,
                                                  gather_solution, gather_ori_solution):
        fw.write(json.dumps(
            {"question": qs, "original_sln": ori_ans, "pred_solution": solution,
             "pred": pred, "label": label},
            ensure_ascii=False) + "\n")


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--test_set", type=str,
                           default="/mnt/Data/qjh/UniTE-main/datasets/TriviaQA/wikipedia-dev-1900.jsonl")
    arg_parse.add_argument("--prompts", type=str,
                           default="/mnt/Data/qjh/UniTE-main/datasets/TriviaQA/prompt.txt")
    arg_parse.add_argument("--model_path1", type=str, default="/mnt/Data/multi-agent/Qwen/Qwen3-4B")
    arg_parse.add_argument("--model_path2", type=str, default="/mnt/Data/multi-agent/Llama-3.2-3B-Instruct")
    arg_parse.add_argument("--output_file", type=str,
                           default="./result_pro_triviaqa.jsonl")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)
    arg_parse.add_argument("--max_new_tokens", type=int, default=256,
                           help="TriviaQA 用'短解释+答案'格式；解释约 30–50 token，"
                                "答案 1–3 token，64 留余量。")
    arg_parse.add_argument("--history_window", type=int, default=2,
                           help="动态权重的滑动窗口大小。TriviaQA 答案短，5 比 20 更敏感。")
    arg_parse.add_argument("--min_weight", type=float, default=0.45,
                           help="任一模型权重的下限 (1-min_weight 为另一模型下限)。")

    args = arg_parse.parse_args()

    accelerator = Accelerator()

    device1 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prompt_complex = open(args.prompts, "r", encoding="utf-8").read()

    model1 = AutoModelForCausalLM.from_pretrained(args.model_path1, device_map=device1,
                                                  torch_dtype=torch.float16).eval()
    model2 = AutoModelForCausalLM.from_pretrained(args.model_path2, device_map=device2,
                                                  torch_dtype=torch.float16).eval()
    tokenizer1, tokenizer2 = (AutoTokenizer.from_pretrained(args.model_path1),
                              AutoTokenizer.from_pretrained(args.model_path2))
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer1.padding_side = "left"
    tokenizer2.padding_side = "left"

    generation_config1 = GenerationConfig(
        num_beams=1, do_sample=False, pad_token_id=tokenizer1.eos_token_id,
        max_new_tokens=1, output_hidden_states=True, output_scores=True,
        output_logits=True, return_dict_in_generate=True, use_cache=True,
    )
    generation_config2 = GenerationConfig(
        num_beams=1, do_sample=False, pad_token_id=tokenizer2.eos_token_id,
        max_new_tokens=1, output_hidden_states=True, output_scores=True,
        output_logits=True, return_dict_in_generate=True, use_cache=True,
    )

    test_dataset = load_dataset("json", data_files=args.test_set)['train']
    ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size,
                           collate_fn=triviaqa_collate_fn, num_workers=2)
    ds_loader = accelerator.prepare_data_loader(ds_loader)

    print('Start ensembling *********************:')
    ensemble_decoding()
    qa_parse_pred_ans(args.output_file)
    print('End ensembling =======================:')
