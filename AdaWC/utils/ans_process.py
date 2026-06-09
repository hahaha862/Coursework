import json
import re

# GSM
def gsm_parse_pred_ans(filename):
    total, correct = 0, 0
    gold_ans = []
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            if jo["original_sln"] not in gold_ans:
                correct += jo["pred"] == jo["label"]
                total += 1
                gold_ans.append(jo["original_sln"])
            else:
                continue
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / total)))

# ARC/PIQA/MMLU
def arc_parse_pred_ans(filename):
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

# SQuAD-style normalization (TriviaQA official preprocessing)
_ARTICLES_RE = re.compile(r'\b(a|an|the)\b')
_PUNCT_RE = re.compile(r'[\u2000-\u206F\u2E00-\u2E7F\'!"#$%&()*+,\-./:;<=>?@\[\]\\^_`{|}~]')

def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = _ARTICLES_RE.sub(' ', s)
    s = _PUNCT_RE.sub(' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

#TriviaQA NQ: SQuAD-style normalized bidirectional substring match.
#TriviaQA 的 label 是 alias 列表（['Sunset Blvd', 'Sunset Boulevard', ...]），
#预测可能是其中任意一个的完整/缩写形式，因此需要双向子串匹配 + 双向 normalize。
def qa_parse_pred_ans(filename):
    total, correct = 0, 0
    with open(filename, "r", encoding="utf-8") as fr:
        for line in fr:
            jo = json.loads(line.strip())
            pred_norm = _normalize_answer(jo["pred"])
            hit = False
            for gold in jo["label"]:
                gold_norm = _normalize_answer(gold)
                if not pred_norm or not gold_norm:
                    continue
                if pred_norm == gold_norm:
                    hit = True
                    break
                # 双向子串：完整/缩写别名都能匹配
                if pred_norm in gold_norm or gold_norm in pred_norm:
                    hit = True
                    break
            if hit:
                correct += 1
            total += 1
    print('num_q %d correct %d ratio %.4f' % (total, correct, float(correct / max(total, 1))))
    return float(correct / max(total, 1))

