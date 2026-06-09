def piqa_collate_fn(batch): #PIQA
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        A = b["A"]
        B = b["B"]
        prompt_q = f'Answer the question by replying A or B.\nQuestion: {ques}\nA: {A}\nB: {B}\nAnswer:'
        questions.append(prompt_q)
        answers.append(b["answer"])
    return questions, answers


def arc_collate_fn(batch): #ARC-C
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        A = b["A"]
        B = b["B"]
        C = b["C"]
        D = b["D"]
        prompt_q = f'Answer the question by replying A, B, C or D.\nQuestion: {ques}\nA: {A}\nB: {B}\nC: {C}\nD: {D}\nAnswer:'
        questions.append(prompt_q)
        answers.append(b["answer"])

    return questions, answers
