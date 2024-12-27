import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--question-file', type=str)
parser.add_argument('--result-file', type=str)
args = parser.parse_args()

questions_file = args.question_file
result_file = args.result_file

questions = [json.loads(q) for q in open(questions_file)]
result_file = [json.loads(q) for q in open(result_file)]

for q, r in zip(questions, result_file):
    if q['question_id'] != r['question_id']:
        print(f'Question ID mismatch: {q["question_id"]} vs {r["question_id"]}')
        
    correct_ans = q['label']
    pred_ans = r['text']
    
    if correct_ans == pred_ans:
        r['correct'] = True
    else:
        r['correct'] = False
        
print(f'Total: {len(questions)}, Correct: {len([q for q in result_file if q["correct"]])}')
print(f'Accuracy: {len([q for q in result_file if q["correct"]]) / len(questions) * 100:.2f}%')
