from collections import defaultdict
import random
import json
import sys
random.seed(42)
DATA_MODEL = sys.argv[1]
MODEL_MAP = {'qwen3-8b': 'qwen3-8b', 'r1-8b-0528': 'r1-8b-0528'}
HF_MODEL_NAME = MODEL_MAP.get(DATA_MODEL, DATA_MODEL)

from datasets import load_dataset
print(f'Loading AISafety-Student/reasoning-safety-behaviours ...')
ds = load_dataset('AISafety-Student/reasoning-safety-behaviours')

rows = []
for split_name in ds:
    for row in ds[split_name]:
        rows.append(row)
        
print(f'Total rows across all splits: {len(rows)}')
sample = rows[0]
print(f'Columns: {list(sample.keys())}')
print(f"Sample 'model' values: {set((str(r.get('model', '')) for r in rows[:500]))}")

data = []
behaviors = defaultdict(list)
prompt_to_indices = defaultdict(list)

for row in rows:
    model_val = str(row.get('model', '')).strip()
    if model_val != HF_MODEL_NAME:
        continue
    labels = row.get('labels')
    if not labels:
        continue
    prompt_val = str(row.get('prompt', '')).strip()
    if not prompt_val:
        continue
        
    target_sentence = str(row.get('target_sentence', '')).strip()
    context = str(row.get('context', '')).strip()
    
    for label in labels:
        label = str(label).strip()
        if not label:
            continue
        d = {'model_name': DATA_MODEL, 'behavior': label, 'query': prompt_val, 'sentence': target_sentence, 'context': context, 'all_labels': labels, 'judge': str(row.get('judge', ''))}
        idx = len(data)
        data.append(d)
        behaviors[label].append(idx)
        prompt_to_indices[prompt_val].append(idx)
        
print(f"Kept {len(data)} entries for model '{HF_MODEL_NAME}' ({DATA_MODEL})")
print(f'Unique behaviors: {len(behaviors)}')

if len(data) == 0:
    avail = set((str(r.get('model', '')) for r in rows[:2000]))
    print(f"ERROR: no data. Available 'model' values: {avail}")
    sys.exit(1)
    
all_prompts = list(prompt_to_indices.keys())
random.shuffle(all_prompts)
split_pt = int(0.8 * len(all_prompts))
train_prompts = set(all_prompts[:split_pt])
test_prompts = set(all_prompts[split_pt:])
train_ids = set()

for pid in train_prompts:
    train_ids.update(prompt_to_indices[pid])
test_ids = set()
for pid in test_prompts:
    test_ids.update(prompt_to_indices[pid])
    
assert train_prompts.isdisjoint(test_prompts)
assert train_ids.isdisjoint(test_ids)

train_split, test_split = ({}, {})
all_idx = list(range(len(data)))

for behavior, pos_list in behaviors.items():
    tr_pos = [i for i in pos_list if i in train_ids]
    te_pos = [i for i in pos_list if i in test_ids]
    tr_neg = [i for i in all_idx if i in train_ids and data[i]['behavior'] != behavior]
    te_neg = [i for i in all_idx if i in test_ids and data[i]['behavior'] != behavior]
    random.shuffle(tr_pos)
    random.shuffle(te_pos)
    random.shuffle(tr_neg)
    random.shuffle(te_neg)
    train_split[behavior] = [{'pos': p, 'neg': random.choice(tr_neg)} for p in tr_pos] if tr_pos and tr_neg else []
    test_split[behavior] = [{'pos': p, 'neg': random.choice(te_neg)} for p in te_pos] if te_pos and te_neg else []
    
OUTPUT = 'data.json'
out = {'data': data, 'train': train_split, 'test': test_split, 'meta': {'dataset': 'AISafety-Student/reasoning-safety-behaviours', 'split_type': 'prompt_level', 'data_model': DATA_MODEL, 'hf_model_name': HF_MODEL_NAME, 'label_field': 'behavior', 'seed': 42, 'num_sentences': len(data), 'num_prompts': len(all_prompts), 'num_train_prompts': len(train_prompts), 'num_test_prompts': len(test_prompts), 'num_train_sentences': len(train_ids), 'num_test_sentences': len(test_ids)}}

with open(OUTPUT, 'w') as f:
    json.dump(out, f)
    
print(f'\nSaved {OUTPUT}  —  {len(behaviors)} behaviors')
print(f'  Prompts : train={len(train_prompts)}, test={len(test_prompts)}')
print(f'  Sentences: train={len(train_ids)}, test={len(test_ids)}\n')

for b, ids in sorted(behaviors.items(), key=lambda x: -len(x[1])):
    tr = sum((1 for i in ids if i in train_ids))
    te = sum((1 for i in ids if i in test_ids))
    print(f'  {len(ids):>5d} total | {tr:>5d} train | {te:>5d} test | {b}')
