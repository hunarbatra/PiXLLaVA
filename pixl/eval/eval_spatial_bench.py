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

tasks = ['counting', 'existence', 'positional', 'reach', 'size']
task_wise_totals = {k: 0 for k in tasks}
task_wise_results = {k: 0 for k in tasks}

for q, r in zip(questions, result_file):
    if q['question_id'] != r['question_id']:
        print(f'Question ID mismatch: {q["question_id"]} vs {r["question_id"]}')
        
    correct_ans = q['answer']
    pred_ans = r['text']
    category = q['category']
    
    task_wise_totals[category] += 1
    
    if correct_ans == pred_ans:
        r['correct'] = True
        task_wise_results[category] += 1
    else:
        r['correct'] = False
        
print(f'Total: {len(questions)}, Correct: {len([q for q in result_file if q["correct"]])}')
print(f'Accuracy: {len([q for q in result_file if q["correct"]]) / len(questions) * 100:.2f}%')

print("-----------------TASK WISE RESULTS------------------")
for task in tasks:
    print(f"{task}: {task_wise_results[task]} / {task_wise_totals[task]} = {task_wise_results[task] / task_wise_totals[task] * 100:.2f}%")
