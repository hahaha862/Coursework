"""unite2_triviaqa.py — TriviaQA 专用单 token ensemble decoding。

相对原 unite2.py 的修复点：
  1. 早停条件从 "<END>" 死代码改为实际可触发的句末/换行/Question: 检测。
  2. collate_fn 中 prompt 拼接改用单换行，避免 few-shot 与当前 question 之间多出空行。
  3. max_new_tokens 默认 64（TriviaQA 用"短解释+答案"格式，单答案 1–3 token 不够）。
  4. 仅服务 TriviaQA，移除 GSM/ARC/PIQA/NQ 等其他数据集的 collate_fn 和路由分支。
  5. 答案抽取：先匹配 "Final answer: X" 行作为 pred，没匹配到再走兜底截断。
  6. 保留动态权重逻辑所需的原始 step 入口（unite2_triviaqa.py 不启用动态权重，pro 版启用）。
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


# ---------------------------------------------------------------------------
# Prompt 拼接
# ---------------------------------------------------------------------------
# few-shot 格式: "Question: ... / Explanation: ... / Final answer: X"
# 当前 question 也按这个格式拼：把模型引到 Explanation: 之后，让它先短解释再给最终答案。
# 这样有两个好处：
#   (a) 强制模型先"想一下"，对 TriviaQA 这种知识题通常能提点。
#   (b) 生成 token 数变长，pro 版的动态权重才有更多步可调（短答案场景下原本 HISTORY_WINDOW 攒不满）。
# 单换行（不是 \n\n），与 few-shot 段间间隔一致。
PROMPT_TEMPLATE = "\nQuestion: {question}\nExplanation:"

# 早停触发：模型在生成完答案后开始"接着写"。
# 不能用 "Explanation:" 当 stop pattern——它是 prompt 的结尾，模型一定会从它后面开始。
# "Final answer:" 也不当 stop pattern——模型要写完它才算完。
# 真正能用的信号：
#   1) "Final answer: <X>\n" 完整一行结束（答案完整）
#   2) "\n\n" 出现（开始新段落）
#   3) "Question:" 出现（开始写下一题）
STOP_STRINGS = ("\n\n", "Question:")
FINAL_ANSWER_LINE_RE = re.compile(r'final answer:\s*([^\n]+)', re.IGNORECASE)


def triviaqa_collate_fn(batch):
    """TriviaQA: prompt_complex + 当前 question + Explanation: 前缀，答案保留为 alias list。"""
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        prompt_q = prompt_complex + PROMPT_TEMPLATE.format(question=ques)
        questions.append(prompt_q)
        answers.append(b["answer"])
    return questions, answers


# ---------------------------------------------------------------------------
# 词汇层工具
# ---------------------------------------------------------------------------
def get_top_k_tokens(outputs, tokenizer, k=10):
    logits = outputs.logits[0]
    probs = logits
    top_k_indices = torch.topk(probs, k).indices
    probs = probs.tolist()

    top_k_probs = []
    for idx, prob in zip(top_k_indices, probs):
        prob_item = [prob[i] for i in idx]
        top_k_probs.append(prob_item)

    top_k_tokens = []
    for indices in top_k_indices:
        token_item = [tokenizer.convert_ids_to_tokens(idx.item(), skip_special_tokens=True) for idx in indices]
        top_k_tokens.append(token_item)

    v1 = []
    for token, prob, id in zip(top_k_tokens, top_k_probs, top_k_indices):
        v1.append(
            {t.replace('▁', 'Ġ').replace('<0x0A>', '/n').replace('Ċ', '/n'): [p, int(i)]
             for t, p, i in zip(token, prob, id)}
        )
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
            # 特殊 token id 兜底
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
                # 空白 token 兜底
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
# 早停判断 + 答案抽取
# ---------------------------------------------------------------------------
def _should_stop(decoded_text: str) -> bool:
    """停止条件（按可触发性排序）：
      1) "Final answer: <X>" 后面已接 \\n——答案行已写完
      2) \\n\\n——开始新段落
      3) Question:——开始写下一题
    注意：Explanation: 不在停止集合里（它是 prompt 结尾，模型必然从这里开始写）。
    """
    if re.search(r'final answer:\s*[^\n]+\n', decoded_text, re.IGNORECASE):
        return True
    for pat in STOP_STRINGS:
        if pat in decoded_text:
            return True
    return False


def _truncate_output(text: str) -> str:
    """从生成文本里抽出 pred 答案。
    优先匹配 "Final answer: X" 这一行；没有则按 Question:/\\n\\n 兜底截断；
    都没有就返回原文本（让判分逻辑自己 SQuAD 归一化处理）。"""
    # 注意：必须先抽 Final answer: X 再做空白归一化——
    # 如果先 normalize whitespace，\n 会变成空格，正则 [^\n]+ 就失去了终止符，
    # 会匹配到 "Triggers\\n\\nQuestion: next" 的全部内容。
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

            # 固定 0.5/0.5 等权融合
            next_token, v_avg, next_token_id1, next_token_id2 = average_and_sample(
                v1_new, v2_new, 0.5, tokenizer1)

            # EOS
            if next_token_id1:
                eos_ids = tokenizer1.eos_token_id
                if not isinstance(eos_ids, list):
                    eos_ids = [eos_ids]
                if next_token_id1[0] in eos_ids:
                    break

            # 句末/换行/Question 早停
            i1, m1 = [], []
            for pred_id, ids, mask in zip(next_token_id1, input_ids1, attention_mask1):
                ids = ids.tolist() + [pred_id]
                mask = mask.tolist() + [1]
                i1.append(ids)
                m1.append(mask)
            input_ids1 = torch.tensor(i1).to(device1)
            attention_mask1 = torch.tensor(m1).to(device1)

            generated_text = tokenizer1.decode(input_ids1[0][input_length[0]:], skip_special_tokens=True)
            if _should_stop(generated_text):
                break

            # 同步到 model2
            iter_input2 = tokenizer2(tokenizer1.batch_decode(input_ids1), padding=True,
                                      return_tensors="pt").to(device2)
            input_ids2 = iter_input2['input_ids'].to(device2)
            attention_mask2 = iter_input2['attention_mask'].to(device2)

        # 解码 + 清理：full_text 留底，pred 抽取
        # 之前 bug：output 立刻被 _truncate_output 覆盖，完整生成文本（含 Explanation）
        # 全部丢失。分开存：
        #   output_ans (per-batch) = 完整生成文本
        #   pred_list  (global)    = 抽取后的最终答案（送进 qa_parse_pred_ans）
        #   solution_list (global) = 完整文本（jsonl 的 pred_solution 字段）
        batch_pred_start = len(pred_list)
        for qs_len, ans in zip(input_length, input_ids1):
            full_text = tokenizer1.decode(ans[qs_len:], skip_special_tokens=True)
            pred = _truncate_output(full_text)
            output_ans.append(full_text)
            pred_list.append(pred)
        solution_list.extend(output_ans)

        # TriviaQA: label 是 alias 列表，直接保留
        label_list.extend(answers)
        ori_ans_list.extend(answers)
        # 打印：本 batch 的完整文本（截 200 字符） + 抽出的 pred
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
                           default="./result_triviaqa.jsonl")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)
    arg_parse.add_argument("--max_new_tokens", type=int, default=256,
                           help="TriviaQA 用'短解释+答案'格式；解释约 30–50 token，答案 1–3 token，"
                                "64 留余量；'Final answer: X\\n' / '\\n\\n' / 'Question:' 都会触发早停。")

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
