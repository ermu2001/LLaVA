import os
import json

out_questions = []
annotations_dir = "playground/data/eval/pope/POPE/output/coco"
out_file = 'llava_pope_test.jsonl'
for file in os.listdir(annotations_dir):
    assert file.startswith('coco_pope_')
    assert file.endswith('.json')
    category = file[10:-5]
    with open(os.path.join(annotations_dir, file)) as f:
        questions = [json.loads(s) for s in f]
    
    for i in range(len(questions)):
        question = questions[i]
        question_id = len(out_questions)
        question['question_id'] = question_id
        question['category'] = category
        out_questions.append(question)
        
with open(out_file, 'w') as f:
    for question in out_questions:
        f.write(json.dumps(question) + "\n")


