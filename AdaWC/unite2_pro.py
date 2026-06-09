import argparse
import json
import re
import os

import torch
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from accelerate.utils import gather_object

from utils.ans_process import *
from utils.extract_response import *


def softmax(x):
  exp_x = np.exp(x)
  sum_exp_x = np.sum(exp_x)
  softmax_x = exp_x / sum_exp_x

  return softmax_x


def qa_collate_fn(batch): #TriviaQA/ NQ
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        prompt_q = prompt_complex + f'\n\nQuestion: {ques}\nAnswer:'
        questions.append(prompt_q)
        answers.append(b["answer"])
    return questions, answers


def gsm_collate_fn(batch): #GSM8K
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        prompt_q = prompt_complex + f'\n\nQuestion: {ques}\nLet\'s think step by step\n'
        questions.append(prompt_q)
        answers.append(b["answer"])
    return questions, answers


def gsm_extract_math_answer(completion):
    ANS_RE = re.compile(r'The answer is\\boxed{(-?\d+)}')
    ANS_RE2 = re.compile(r'The answer is \\\\boxed{(-?\d+)}')
    ANS_RE3 = re.compile(r'Answer: \\\\boxed{(-?\d+)}')
    ANS_RE4 = re.compile(r'answer is \\boxed{(-?\d+)}')
    ANS_RE5 = re.compile(r"#### (-?\d+)")
    ANS_RE6 = re.compile(r"answer is (-?\d+)")
    ANS_RE7 = re.compile(r"answer: (-?\d+)")
    ANS_RE8 = re.compile(r'\\boxed{(-?\d+)}')
    ANS_RE9 = re.compile(r'答案为\$\{\\boxed{(-?\d+)\}\}\$')
    ANS_RE10 = re.compile(r'答案为\$\\boxed{(-?\d+)}\$')

    for ans_re in [ANS_RE, ANS_RE2, ANS_RE3, ANS_RE4, ANS_RE5, ANS_RE6, ANS_RE7, ANS_RE8, ANS_RE9, ANS_RE10]:
        match = ans_re.search(completion)
        if match:
            match_str = match.group(1).strip()
            # match_str = match_str.replace(",", "")
            try:
                return float(match_str)
            except:
                continue

    match = re.search(r'(-?\d+)\.$', completion)
    if match:
        match_str = match.group(1).strip()
        return float(match_str)
    else:
        return 0


def get_top_k_tokens(outputs, tokenizer, k=10):
    logits = outputs.logits[0]
    probs = logits

    top_k_indices = torch.topk(probs, k).indices
    probs = probs.tolist()

    top_k_probs = []
    for idx, prob in zip(top_k_indices,probs):
        prob_item = []
        for i in idx:
            prob_item.append(prob[i])
        top_k_probs.append(prob_item)

    top_k_tokens = []
    for indices in top_k_indices:
        token_item = []
        for idx in indices:
            token_item.append(tokenizer.convert_ids_to_tokens(idx.item(), skip_special_tokens=True))
        top_k_tokens.append(token_item)


    v1 = []
    for token, prob, id in zip(top_k_tokens, top_k_probs, top_k_indices):
        v1.append(
            {token.replace('▁','Ġ').replace('<0x0A>','/n').replace('Ċ','/n'): [prob, int(id)] for token, prob, id in zip(token, prob, id)})

    return v1


def get_union_vocab(v1, v2):
    v_all = []
    for i in range(len(v1)):
        v1_token = v1[i].keys()
        v2_token = v2[i].keys()
        unique_tokens = list(set(v1_token).union(set(v2_token)))
        v_all.append(unique_tokens)
    return v_all


def update_vocab(v1, vu, tokenizer, logits, model_name):
    for vu_token, v1_token, logit_ele in zip(vu,v1,logits):
        v1_token_ids = []
        for item in v1_token.values():
            v1_token_ids.append(item[1])
        for token in vu_token:
            if token not in v1_token.keys():
              #Take special token id into consideration
              if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                  token = token.replace('Ġ', '▁')
              if token != '':
                  subtoken_id = tokenizer.convert_tokens_to_ids(token)
                  if subtoken_id != 0 and subtoken_id != None: #Mistral and Llama2 oov id 0
                      logit = logit_ele[subtoken_id]
                  else:
                      subtokens = tokenizer.tokenize(token)
                      for token_id in tokenizer.convert_tokens_to_ids(subtokens):
                          if 'llama2' in model_name:
                              if token_id != 29871:
                                  subtoken_id = token_id
                                  break
                          if 'mistral' in model_name:
                              if token_id != 29473:
                                  subtoken_id = token_id
                                  break
                          if 'deepseek' in model_name:
                              if token_id != 207:
                                  subtoken_id = token_id
                                  break
                          if 'openchat' in model_name:
                              if token_id != 28705:
                                  subtoken_id = token_id
                                  break
                          else:
                              subtoken_id = token_id
                              break
                      logit = logit_ele[subtoken_id]
              else:
                  if 'llama3' in model_name or 'qwen2' in model_name:
                      logit = logit_ele[220]
                      subtoken_id = 220
                  if 'llama2' in model_name:
                      logit = logit_ele[29871]
                      subtoken_id = 29871
                  if 'mistral' in model_name:
                      logit = logit_ele[29473]
                      subtoken_id = 29473
                  if 'deepseek' in model_name:
                      logit = logit_ele[207]
                      subtoken_id = 207
                  if 'openchat' in model_name:
                      logit = logit_ele[28705]
                      subtoken_id = 28705
                  if 'glm' in model_name:
                      logit = logit_ele[128]
                      subtoken_id = 128

              if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                  v1_token[token.replace('▁', 'Ġ')] = [logit, subtoken_id]
              else:
                if subtoken_id not in v1_token_ids:
                    v1_token[token] = [logit, subtoken_id]
                    v1_token_ids.append(subtoken_id)
                else:
                    v1_token[token] = [0, subtoken_id]

    v1_new = vocab_softmax(v1)
    return v1_new


def vocab_softmax(v1):
    v1_new = []
    for element in v1:
        ele = {}
        ele_values = list(element.values())
        ele_values0, ele_values1 = [], []
        for item in ele_values:
            ele_values0.append(item[0])
            ele_values1.append(item[1])
        ele_values0 = torch.softmax(torch.tensor(ele_values0), dim=0)
        for token, prob, ids in zip(element.keys(),ele_values0,ele_values1):
          ele[token] = [prob, ids]
        v1_new.append(ele)

    return v1_new


def drop_token(v1,v2,t):
    v1_new, v2_new = [], []
    for v1_element, v2_element in zip(v1,v2):
        v1_, v2_ = {}, {}
        for key in v1_element.keys():
            if v1_element[key][0] > t:
                v1_[key] = v1_element[key]
                v2_[key] = v2_element[key]
        v1_new.append(v1_)
        v2_new.append(v2_)
    return v1_new,v2_new


def average_and_sample(v1, v2, lamda, tokenizer):
    next_token, v_avg, next_token_id1,next_token_id2 = [], [], [], []
    for element_v1, element_v2 in zip(v1,v2):
        assert len(element_v1) == len(element_v2)
        v_new = {}
        for token1 in element_v1:
            v_new[token1] = [lamda * element_v1[token1][0] + (1-lamda) * element_v2[token1][0],element_v1[token1][1]]
        v_avg.append(v_new)

        probs = []
        for item in v_new.values():
            probs.append(item[0])


        sample_index = probs.index(max(probs))

        i = 0
        for item1 in v_new.keys():
            if i == sample_index:
                next_token.append(tokenizer.convert_ids_to_tokens(element_v1[item1][1]))
                next_token_id1.append(element_v1[item1][1])
                next_token_id2.append(element_v2[item1][1])
            i+=1

    return next_token, v_avg, next_token_id1, next_token_id2


def pad_list(list_name,pad_id):
    list_len = [len(item) for item in list_name]
    max_len = max(list_len)
    for item in list_name:
        if len(item) < max_len:
            pad = [pad_id] * (max_len - len(item))
            pad.extend(item)
            item[:] = pad

    return list_name


def ensemble_decoding(test):
    fw = open(args.output_file, "w", encoding="utf-8")

    accelerator.wait_for_everyone()
    solution_list, pred_list, label_list, ori_ans_list, question_list = [], [], [], [], []

    if accelerator.is_main_process:
        iter_item = tqdm(ds_loader)
    else:
        iter_item = ds_loader


    max_length = args.max_new_tokens

    # ====================== 改进：动态权重初始化 ======================
    HISTORY_WINDOW = 20
    MIN_WEIGHT = 0.2
    model1_score_history = []
    model2_score_history = []
    # ====================== 改进：动态权重初始化 ======================

    for questions, answers in iter_item:
        output_ans = []

        inputs1 = tokenizer1(questions, padding=True, return_tensors="pt").to(device1)
        inputs2 = tokenizer2(questions, padding=True, return_tensors="pt").to(device2)
        input_ids1 = inputs1['input_ids'].to(device1)
        input_ids2 = inputs2['input_ids'].to(device2)

        attention_mask1 = inputs1['attention_mask'].to(device1)
        attention_mask2 = inputs2['attention_mask'].to(device2)

        input_length = [len(qs) for qs in input_ids1]

        distribution1, distribution2 = [], []
        for i in range(max_length):
            if i == 0: #first step
                outputs1 = model1.generate(input_ids=input_ids1,
                                           attention_mask=attention_mask1,
                                           generation_config=generation_config1,
                                           )
                outputs2 = model2.generate(input_ids=input_ids2,
                                           attention_mask=attention_mask2,
                                           generation_config=generation_config2,
                                           )

            else:
                outputs1 = model1.generate(input_ids=input_ids1,
                                           attention_mask=attention_mask1,
                                           past_key_values=past_key_values1,
                                           generation_config=generation_config1,
                                           )
                outputs2 = model2.generate(input_ids=input_ids2,
                                           attention_mask=attention_mask2,
                                           generation_config=generation_config2,
                                           )



            past_key_values1 = outputs1.past_key_values


            logits1 = torch.max(torch.softmax(torch.topk(outputs1.logits[0][0], 10).values, dim=0)).item()
            logits2 = torch.max(torch.softmax(torch.topk(outputs2.logits[0][0], 10).values, dim=0)).item()

            distribution1.append(logits1)
            distribution2.append(logits2)


            v1 = get_top_k_tokens(outputs1,tokenizer1,10)
            v2 = get_top_k_tokens(outputs2,tokenizer2,10)

            v1_sfmx = vocab_softmax(v1)
            v2_sfmx = vocab_softmax(v2)

            vu = get_union_vocab(v1, v2)

            v1_update = update_vocab(v1, vu, tokenizer1, outputs1.logits[0],'qwen2')
            v2_update = update_vocab(v2, vu, tokenizer2, outputs2.logits[0],'llama3')

            v1_new, v2_new = v1_update, v2_update

            # ====================== 改进1：冗余融合跳过 ======================
            # 检查两个模型的 top-1 token 是否一致，一致则直接用，跳过融合
            skip_fusion = False
            if len(v1_new) > 0 and len(v2_new) > 0 and len(model1_score_history) > 0:
                v1_dict = v1_new[0]
                v2_dict = v2_new[0]
                if v1_dict and v2_dict:
                    # 获取两个模型的 top-1 token
                    v1_sorted = sorted(v1_dict.items(), key=lambda x: x[1][0], reverse=True)
                    v2_sorted = sorted(v2_dict.items(), key=lambda x: x[1][0], reverse=True)
                    if v1_sorted and v2_sorted:
                        v1_top1_token = v1_sorted[0][0]
                        v2_top1_token = v2_sorted[0][0]
                        if v1_top1_token == v2_top1_token:
                            skip_fusion = True
            # ====================== 改进1：冗余融合跳过 ======================

            # ====================== 改进2：动态权重计算 ======================
            if len(model1_score_history) > 0:
                avg_score1 = sum(model1_score_history) / len(model1_score_history)
                avg_score2 = sum(model2_score_history) / len(model2_score_history)
                total_score = avg_score1 + avg_score2 + 1e-8
                w1 = avg_score1 / total_score
                w1 = max(MIN_WEIGHT, min(1 - MIN_WEIGHT, w1))
                w2 = 1 - w1
            else:
                w1, w2 = 0.5, 0.5
            # ====================== 改进2：动态权重计算 ======================

            # 如果跳过融合，直接用任一模型的 top-1
            if skip_fusion:
                v1_dict = v1_new[0]
                v2_dict = v2_new[0]
                v1_sorted = sorted(v1_dict.items(), key=lambda x: x[1][0], reverse=True)
                v2_sorted = sorted(v2_dict.items(), key=lambda x: x[1][0], reverse=True)
                next_token, v_avg, next_token_id1, next_token_id2 = [], [], [], []
                top_token = v1_sorted[0][0]
                next_token.append(tokenizer1.convert_ids_to_tokens(v1_dict[top_token][1]))
                next_token_id1.append(v1_dict[top_token][1])
                next_token_id2.append(v2_dict[top_token][1])
                # 构造 v_avg 用于历史分数更新
                v_avg_dict = {}
                for token in v1_dict:
                    v_avg_dict[token] = [w1 * v1_dict[token][0] + w2 * v2_dict[token][0], v1_dict[token][1]]
                v_avg.append(v_avg_dict)
            else:
                next_token, v_avg, next_token_id1, next_token_id2 = average_and_sample(v1_new, v2_new, w1, tokenizer1)

            # ====================== 改进2：更新历史分数 ======================
            if len(v_avg) > 0 and len(next_token_id1) > 0:
                final_token_str = next_token[0] if len(next_token) > 0 else None
                if final_token_str and final_token_str in v1_new[0] and final_token_str in v2_new[0]:
                    score1 = v1_new[0][final_token_str][0]
                    score2 = v2_new[0][final_token_str][0]
                    model1_score_history.append(score1)
                    model2_score_history.append(score2)
                    model1_score_history = model1_score_history[-HISTORY_WINDOW:]
                    model2_score_history = model2_score_history[-HISTORY_WINDOW:]
            # ====================== 改进2：更新历史分数 ======================

            # Early stopping if EOS token is generated (check all possible EOS tokens)
            if next_token_id1:
                eos_ids = tokenizer1.eos_token_id
                if not isinstance(eos_ids, list):
                    eos_ids = [eos_ids]
                if next_token_id1[0] in eos_ids:
                    print(f"EOS token generated, stopping early at step {i}")
                    break

            # Early stopping if END marker token is generated
            if next_token and "<END>" in next_token[0]:
                print(f"END marker token generated, stopping early at step {i}")
                break

            i1, i2, m1, m2 = [], [], [], []
            for pred_token_id1, pred_token_id2, input1_ids, input2_ids, mask1, mask2 in zip(next_token_id1,next_token_id2,input_ids1,input_ids2,attention_mask1,attention_mask2):
                input1_ids = input1_ids.tolist()
                mask1 = mask1.tolist()
                input1_ids.append(pred_token_id1)
                mask1.append(1)
                i1.append(input1_ids)
                m1.append(mask1)

            input_ids1 = torch.tensor(i1).to(device1)
            attention_mask1 = torch.tensor(m1).to(device1)

            # Only check newly generated text (not the original prompt)
            generated_text = tokenizer1.decode(input_ids1[0][input_length[0]:], skip_special_tokens=True)
            if "<END>" in generated_text:
                print(f"END marker found in generated text, stopping early at step {i}")
                break


            iter_input2 = tokenizer2(tokenizer1.batch_decode(input_ids1), padding=True, return_tensors="pt").to(device2)

            input_ids2 = iter_input2['input_ids'].to(device2)
            attention_mask2 = iter_input2['attention_mask'].to(device2)


        for qs_len, ans in zip(input_length, input_ids1):
            output = tokenizer1.decode(ans[qs_len:], skip_special_tokens=True)
            output = ' '.join(output.split())
            # Truncate output at END marker if present
            if "<END>" in output:
                output = output.split("<END>")[0].strip()
            output_ans.append(output)

        ans_num = []
        for gold_ans in answers:
            if 'gsm' in test:
                match = re.search(r"#### (-?\d+)", gold_ans)
                if not match:
                    match = re.search(r"answer is (-?\d+)", gold_ans, re.IGNORECASE)
                if match:
                    ans_num.append(float(match.group(1)))
                else:
                    ans_num.append(0.0)
            else:
                ans_num.append(gold_ans)
        label_list.extend(ans_num)
        ori_ans_list.extend(answers)

        pred_num = []
        ans_list = []
        for gold_ans in output_ans:
            print(gold_ans)
            if 'Question' in gold_ans:
                gold_ans = gold_ans.split('Question:')[0].strip()
            if 'Explanation' in gold_ans:
                gold_ans = gold_ans.split('Explanation')[0].strip()
            ans_list.append(gold_ans)
            if 'gsm' in test.lower():
                pred_num.append(gsm_extract_math_answer(gold_ans))
            else:
                pred_num.append(gold_ans)
            print('==========output========\n', ans_num[-1],"=======",pred_num[-1])
        pred_list.extend(pred_num)
        solution_list.extend(ans_list)
        question_list.extend(questions)


    accelerator.print("======= waiting for everyone ==========")
    accelerator.wait_for_everyone()
    accelerator.print("======= start gather ==========")
    gather_pred = gather_object(pred_list)
    gather_label = gather_object(label_list)
    gather_solution = gather_object(solution_list)
    gather_ori_solution = gather_object(ori_ans_list)
    gather_qs = gather_object(question_list)

    for qs, pred, label, solution, ori_ans in zip(gather_qs, gather_pred, gather_label, gather_solution,
                                                  gather_ori_solution):
        fw.write(json.dumps(
            {"question": qs, "original_sln": ori_ans, "pred_solution": solution, "pred": pred, "label": label},
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
    arg_parse.add_argument("--max_new_tokens", type=int, default=20)
    arg_parse.add_argument("--history_window", type=int, default=2, help="History window for dynamic weight")
    arg_parse.add_argument("--min_weight", type=float, default=0.2, help="Minimum weight for each model")

    args = arg_parse.parse_args()


    accelerator = Accelerator()

    # load device, prompt
    device1 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    prompt_complex = open(args.prompts, "r", encoding="utf-8").read()

    #load model, tokenizer, generation_config
    model_path1, model_path2= args.model_path1, args.model_path2

    model1 = AutoModelForCausalLM.from_pretrained(model_path1, device_map=device1,
                                      torch_dtype=torch.float16).eval()
    tokenizer1 = AutoTokenizer.from_pretrained(model_path1)
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer1.padding_side = "left"
    generation_config1 = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer1.eos_token_id,
        max_new_tokens=1,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    model2 = AutoModelForCausalLM.from_pretrained(model_path2, device_map=device2,
                                      torch_dtype=torch.float16).eval()
    tokenizer2 = AutoTokenizer.from_pretrained(model_path2)
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer2.padding_side = "left"
    generation_config2 = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer2.eos_token_id,
        max_new_tokens=1,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
    )
    #强制 deterministic，防止模型默认配置覆盖
    model1.generation_config.do_sample = False
    model2.generation_config.do_sample = False

    #load dataset
    if 'GSM' in args.test_set:
        dataset = load_dataset('json', data_files=args.test_set)
        ds_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=args.per_device_batch_size,
                                          collate_fn=gsm_collate_fn)
    else:
        dataset = load_dataset('json', data_files=args.test_set)
        ds_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=args.per_device_batch_size,
                                          collate_fn=qa_collate_fn)
    # accelerator prepare
    ds_loader = accelerator.prepare(ds_loader)

    #infer
    ensemble_decoding(args.test_set.lower())

    # 准确率统计
    if 'gsm' in args.test_set.lower():
        gsm_parse_pred_ans(args.output_file)
    if 'triviaqa' in args.test_set.lower() or 'nq' in args.test_set.lower():
        qa_parse_pred_ans(args.output_file)
    if 'arc' in args.test_set.lower() or 'piqa' in args.test_set.lower():
        arc_parse_pred_ans(args.output_file)
    print('End ensembling =======================:')

