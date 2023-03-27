import os
import json
import random


random.seed(42)


with open(os.path.join('outputs', 'labels_validation.jsonl')) as f:
    labels = [json.loads(line)['code'] for line in f.readlines()]

with open(os.path.join('outputs', 'predictions_validation.txt')) as f:
    preds = f.read().strip().split('\n')

errors = [(label, pred) for label, pred in zip(labels, preds) if label != pred]

selected = random.sample(errors, k=100)
for label, pred in selected:
    print('true', label, sep='\t')
    print('pred', pred, sep='\t')
    print()

print('total errors', len(errors))

shorter = sum(1 for label, pred in errors if len(pred) <= len(label) / 3)
print('3x shorter seqs', shorter, round(shorter / len(errors), 3))

loops = sum(1 for label, pred in errors if pred != label and len(pred) == 511)
print('loops', loops, round(loops / len(errors), 3))
