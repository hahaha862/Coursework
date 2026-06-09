"""unite2_pro_mmlu.py — MMLU 专用 pro 版 ensemble decoding。

相对原 unite2_pro.py 的修复点（与 unite2_pro_triviaqa.py 共享 + MMLU 特有）：
  共享：
    1. 早停条件改为可触发的 "Answer: <X>\\n" / \\n\\n / Question: 检测。
    2. collate_fn 拼接改用单换行；模板结尾 "Explanation:"，先 CoT 思考再给 "Answer: X"。
    3. max_new_tokens 默认 256（CoT 解释约 30-150 token）。
    4. 仅 MMLU，移除其他数据集路由。
  pro 特有：
    5. 动态权重的 HISTORY_WINDOW 默认 2；冷启动修复（history<WINDOW 不裁剪）。
    6. 冗余融合跳过：top-1 token 一致时跳过融合。
  MMLU 特有：
    7. CSV 加载 + ast.literal_eval 解析 choices。
    8. 答案抽取 "Answer: X" 行；label 是 0-3 整数。
"""

from tqdm import tqdm
import re
import json
import ast

import torch
import argparse

from accelerate import Accelerator
from torch.utils.data import DataLoader
from accelerate.utils import gather_object

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset


PROMPT_TEMPLATE = "\nQuestion: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nExplanation:"
STOP_STRINGS = ("\n\n", "Question:")
ANSWER_LINE_RE = re.compile(r'answer:\s*([A-Da-d])\b', re.IGNORECASE)


def mmlu_collate_fn(batch):
    """MMLU: 解析 choices 列表，按 A/B/C/D 排版。"""
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        choices = b["choices"]
        if isinstance(choices, str):
            choices = ast.literal_eval(choices)
        A, B, C, D = choices[0], choices[1], choices[2], choices[3]
        prompt_q = prompt_complex + PROMPT_TEMPLATE.format(
            question=ques, A=A, B=B, C=C, D=D
        )
        questions.append(prompt_q)
        answers.append(b["answer"])
    return questions, answers


# ---------------------------------------------------------------------------
# 词汇层工具
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
    """CoT 早停："Answer: X\n" 完整行结束 / 新段落 / 新题 三选一即停。
    注意：Explanation: 不在停止集合里（它是 prompt 结尾）。"""
    if re.search(r'answer:\s*[A-Da-d][^\n]*\n', decoded_text, re.IGNORECASE):
        return True
    for pat in STOP_STRINGS:
        if pat in decoded_text:
            return True
    return False


def _truncate_output(text: str) -> str:
    """从生成文本里抽 A/B/C/D（CoT 模式下 Explanation 段很长）。"""
    m = re.search(r'answer:\s*([A-Da-d])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    matches = re.findall(r'([A-Da-d])\b', text)
    if matches:
        return matches[-1].upper()
    return text.strip()


def parse_pred_ans(filename):
    """MMLU: pred 是 A/B/C/D，label 是 0-3 整数；统一成大写字母后比较。"""
    total, correct = 0, 0
    seen_qs = []
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            if jo["question"] in seen_qs:
                continue
            seen_qs.append(jo["question"])
            pred = str(jo["pred"]).strip().upper()
            try:
                label_int = int(jo["label"])
                label = "ABCD"[label_int]
            except (ValueError, TypeError, IndexError):
                label = str(jo["label"]).strip().upper()
            if pred == label:
                correct += 1
            total += 1
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / max(total, 1))))
    return float(correct / max(total, 1))


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
            if len(model1_score_history) > 0:
                avg_score1 = sum(model1_score_history) / len(model1_score_history)
                avg_score2 = sum(model2_score_history) / len(model2_score_history)
                total_score = avg_score1 + avg_score2 + 1e-8
                w1 = avg_score1 / total_score
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

            # ---- 推进 + 早停 ----
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

        # 解码 + 清理
        batch_pred_start = len(pred_list)
        for qs_len, ans in zip(input_length, input_ids1):
            full_text = tokenizer1.decode(ans[qs_len:], skip_special_tokens=True)
            pred = _truncate_output(full_text)
            output_ans.append(full_text)
            pred_list.append(pred)
        solution_list.extend(output_ans)

        # MMLU: label 是 0-3 整数，直接保留
        label_list.extend(answers)
        ori_ans_list.extend(answers)
        for ans_label, full, pred in zip(answers, output_ans, pred_list[batch_pred_start:]):
            raw = full[:200].replace('\n', ' ⏎ ')
            print('---raw output---\n' + raw + ('...' if len(full) > 200 else ''))
            print('---extracted pred---:', pred)
            try:
                gold_letter = "ABCD"[int(ans_label)]
            except (ValueError, TypeError, IndexError):
                gold_letter = str(ans_label)
            print('==========output========\n', f'pred={pred} gold={gold_letter}')
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
                           default="/mnt/Data/qjh/UniTE-main/datasets/MMLU/mmlu-all-test.csv")
    arg_parse.add_argument("--prompts", type=str,
                           default="/mnt/Data/qjh/UniTE-main/datasets/MMLU/prompt.txt")
    arg_parse.add_argument("--model_path1", type=str, default="/mnt/Data/multi-agent/Qwen/Qwen3-4B")
    arg_parse.add_argument("--model_path2", type=str, default="/mnt/Data/multi-agent/Llama-3.2-3B-Instruct")
    arg_parse.add_argument("--output_file", type=str,
                           default="./result_pro_mmlu.jsonl")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)
    arg_parse.add_argument("--max_new_tokens", type=int, default=256,
                           help="MMLU CoT：解释约 30-150 token，答案 1-3 token；256 留余量。")
    arg_parse.add_argument("--history_window", type=int, default=2,
                           help="动态权重的滑动窗口大小。MMLU CoT 步数多，窗口 2 即可。")
    arg_parse.add_argument("--min_weight", type=float, default=0.45,
                           help="任一模型权重的下限。")

    args = arg_parse.parse_args()

    accelerator = Accelerator()

    device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    prompt_complex = open(args.prompts, "r", encoding="utf-8").read()

    model1 = AutoModelForCausalLM.from_pretrained(args.model_path1, device_map=device1,
                                                  torch_dtype=torch.float16).eval()
    tokenizer1 = AutoTokenizer.from_pretrained(args.model_path1)
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer1.padding_side = "left"
    generation_config1 = GenerationConfig(
        num_beams=1, do_sample=False, pad_token_id=tokenizer1.eos_token_id,
        max_new_tokens=1, output_hidden_states=True, output_scores=True,
        output_logits=True, return_dict_in_generate=True, use_cache=True,
    )

    model2 = AutoModelForCausalLM.from_pretrained(args.model_path2, device_map=device2,
                                                  torch_dtype=torch.float16).eval()
    tokenizer2 = AutoTokenizer.from_pretrained(args.model_path2)
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer2.padding_side = "left"
    generation_config2 = GenerationConfig(
        num_beams=1, do_sample=False, pad_token_id=tokenizer2.eos_token_id,
        max_new_tokens=1, output_hidden_states=True, output_scores=True,
        output_logits=True, return_dict_in_generate=True, use_cache=True,
    )
    # 强制 deterministic
    model1.generation_config.do_sample = False
    model2.generation_config.do_sample = False

    test_dataset = load_dataset("csv", data_files=args.test_set)['train']
    ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size,
                           collate_fn=mmlu_collate_fn, num_workers=2)
    ds_loader = accelerator.prepare(ds_loader)

    print('Start ensembling *********************:')
    ensemble_decoding()
    parse_pred_ans(args.output_file)
    print('End ensembling =======================:')
