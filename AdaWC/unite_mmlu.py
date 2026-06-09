from tqdm import tqdm
import os
import json
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

import torch
import argparse

from accelerate import Accelerator
from torch.utils.data import DataLoader
from accelerate.utils import gather_object


def extract_math_answer(pred_str): #MMLU
    try:
        if( 'boxed' in pred_str):
            ans = pred_str.split('boxed')[-1]
            flag = 1
        elif('the answer is ' in pred_str):
            ans = pred_str.split('the answer is ')[-1].strip()
            flag = 1
        elif 'The answer is ' in pred_str:
            ans = pred_str.split('The answer is ')[-1].strip()
            flag = 1
        else:
            ans = pred_str
            flag = 0

        pattern = r'[A-D]'
        pred = re.findall(pattern, ans)

        if(len(pred) >= 1):
            # print(pred_str)
            if flag == 0:
                pred = float(pred[-1])
            else:
                pred = float(pred[0])
        else:
            pred = float("nan")


    except Exception:
        print(f"Cannot parse the resulting num in predicted solution {pred_str}.\n")
        pred = float("nan")

    return pred

def collate_fn(batch): #MMLU
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        A = b["A"]
        B = b["B"]
        C = b["C"]
        D = b["D"]
        prompt_q = prompt + f'Answer the question by replying A, B, C or D.\nQuestion: {ques}\nA: {A}\nB: {B}\nC: {C}\nD: {D}\nAnswer:'
        # print('******prompt_q:\n', prompt_q)
        questions.append(prompt_q)
        answers.append(b["answer"])

    return questions, answers

def  parse_pred_ans(filename):
    total, correct = 0, 0
    gold_ans = []
    qs = []
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            if jo["question"] not in qs:
                correct += jo["pred"].strip() == jo["label"].strip()
                total += 1
                qs.append(jo["question"])
            else:
                continue
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))
    return float(correct / total)


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
    # Extract unique tokens from both dictionaries
    unique_tokens = []
    for v1_tokens, v2_tokens in zip(v1,v2):
        unique_tokens.append(list(set(v1_tokens.keys()) | set(v2_tokens.keys())))

    return unique_tokens

def update_vocab(v1, vu, tokenizer, logits, model_name):
    for vu_token, v1_token, logit_ele in zip(vu,v1,logits):
        for token in vu_token:
            if token not in v1_token.keys():
              if 'llama2' in model_name or 'mistral' in model_name:
                  token.replace('Ġ','▁')
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
              # 将{token: logit}添加到v1中
              v1_token[token.replace('▁','Ġ')] = [logit,subtoken_id]

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
    fw = open(args.output_file, "a", encoding="utf-8")

    accelerator.wait_for_everyone()
    solution_list, pred_list, label_list, ori_ans_list, question_list = [], [], [], [], []

    gsm8k_dataset = load_dataset("json", data_files=test)['train']

    ds_loader = DataLoader(gsm8k_dataset, batch_size=args.per_device_batch_size, collate_fn=collate_fn, num_workers=1)
    ds_loader = accelerator.prepare_data_loader(ds_loader)

    if accelerator.is_main_process:
        iter_item = tqdm(ds_loader)
    else:
        iter_item = ds_loader

    # iter_item = ds_loader

    max_length = args.max_new_tokens
    for questions, answers in iter_item:
        output_ans = []
        # inputs_originial = questions
        # flag = 0

        inputs1 = tokenizer1(questions, padding=True, return_tensors="pt").to(device1)
        inputs2 = tokenizer2(questions, padding=True, return_tensors="pt").to(device2)

        input_ids1 = inputs1['input_ids'].to(device1)
        input_ids2 = inputs2['input_ids'].to(device2)

        attention_mask1 = inputs1['attention_mask'].to(device1)
        attention_mask2 = inputs2['attention_mask'].to(device2)

        input_length = [len(qs) for qs in input_ids1]


        for i in range(max_length):
            start_time = time.time()
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


            v1 = get_top_k_tokens(outputs1,tokenizer1,10)
            v2 = get_top_k_tokens(outputs2,tokenizer2,10)

            v1_sfmx = vocab_softmax(v1)
            v2_sfmx = vocab_softmax(v2)

            vu = get_union_vocab(v1, v2)

            v1_update = update_vocab(v1, vu, tokenizer1, outputs1.logits[0],'llama2')
            v2_update = update_vocab(v2, vu, tokenizer2, outputs2.logits[0],'deepseek')

            v1_new, v2_new = v1_update, v2_update

            next_token, v_avg, next_token_id1, next_token_id2 = average_and_sample(v1_new,v2_new,0.5, tokenizer1)

            end_time = time.time()

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

            iter_input2 = tokenizer2(tokenizer1.batch_decode(input_ids1), padding=True, return_tensors="pt").to(device2)

            input_ids2 = iter_input2['input_ids'].to(device2)
            attention_mask2 = iter_input2['attention_mask'].to(device2)

        for qs_len, ans in zip(input_length, input_ids1):
            output = tokenizer1.decode(ans[qs_len:], skip_special_tokens=True)
            output = ' '.join(output.split())
            output_ans.append(output)

        ans_num = []
        for gold_ans in answers:
            ans_num.append(gold_ans)
        label_list.extend(ans_num)
        ori_ans_list.extend(answers)

        pred_num = []
        ans_list = []
        for gold_ans in output_ans:
            if 'Question' in gold_ans:
                gold_ans = gold_ans.split('Question:')[0].strip()
            ans_list.append(gold_ans)
            print('==========output========\n', gold_ans)
            pred_num.append(gold_ans)

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

    for qs, pred, label, solution, ori_ans in zip(gather_qs, gather_pred, gather_label, gather_solution, gather_ori_solution):
        fw.write(json.dumps({"question": qs,"original_sln": ori_ans, "pred_solution": solution, "pred": pred, "label": label},
                            ensure_ascii=False) + "\n")


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--test_set", type=str,
                           default="MMLU/test-jsonl/")
    arg_parse.add_argument("--prompts", type=str,
                           default="MMLU/dev-jsonl/")
    arg_parse.add_argument("--model_path1", type=str, default="Your model path")
    arg_parse.add_argument("--model_path2", type=str, default="Your model path")
    arg_parse.add_argument("--output_file", type=str,
                           default="Your output path")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)

    arg_parse.add_argument("--max_new_tokens", type=int, default=1)

    args = arg_parse.parse_args()

    accelerator = Accelerator()

    # load device, prompt
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    #load model, tokenizer, generation_config
    model_path1, model_path2= args.model_path1, args.model_path2

    model1 = AutoModelForCausalLM.from_pretrained(model_path1, device_map=device1,
                                       attn_implementation="flash_attention_2",
                                       torch_dtype=torch.float16).eval()


    model2 = AutoModelForCausalLM.from_pretrained(model_path2, device_map=device2,
                                       attn_implementation="flash_attention_2",
                                       torch_dtype=torch.float16).eval()

    tokenizer1, tokenizer2 = AutoTokenizer.from_pretrained(model_path1), AutoTokenizer.from_pretrained(model_path2)
    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2.pad_token = tokenizer2.eos_token

    tokenizer1.padding_side = "left"
    tokenizer2.padding_side = "left"

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

    test_files = os.listdir(args.test_set)
    test_file_paths = [os.path.join(args.test_set, file) for file in test_files if
                       os.path.isfile(os.path.join(args.test_set, file))]

    prompt_files = os.listdir(args.prompts)
    prompt_file_paths = [os.path.join(args.prompts, file) for file in prompt_files if
                         os.path.isfile(os.path.join(args.prompts, file))]

    # Initialize the accuracy list
    acc_list = []

    # Create a mapping of test file names without extensions to their paths
    test_file_map = {os.path.splitext(os.path.basename(test_file))[0]: test_file for test_file in test_file_paths}

    # Iterate through prompt files and find corresponding test files
    for promptf in prompt_file_paths:
        prompt_file_name = os.path.splitext(os.path.basename(promptf))[0]

        if prompt_file_name in test_file_map:
            print(prompt_file_name)
            test_file_path = test_file_map[prompt_file_name]
            prompt = ''
            prompt_read = load_dataset("json", data_files=promptf)['train']
            for data in prompt_read:
                prompt += 'Question: ' + data['question'] + '\n'
                prompt += 'A: ' + data['A'] + '\n'
                prompt += 'B: ' + data['B'] + '\n'
                prompt += 'C: ' + data['C'] + '\n'
                prompt += 'D: ' + data['D'] + '\n'
                prompt += 'Answer: ' + data['answer'] + '\n\n'

            print('Start reasoning *********************:')
            ensemble_decoding(test_file_path)
            acc = parse_pred_ans(args.output_file)
            acc_list.append(acc)
            print('End reasoning =======================:')

    print("The avg acc is: ", sum(acc_list) / len(acc_list))
