from collections import defaultdict
import random, json, sys
import pandas as pd
from datasets import load_dataset

random.seed(42)
DATA_MODEL = sys.argv[1]
print('Loading ishitakakkar-10/HarmThoughts from HuggingFace ...')
ds = load_dataset('ishitakakkar-10/HarmThoughts')
frames = []

for split_name in ds:
    frames.append(ds[split_name].to_pandas())
    print(f"  Split '{split_name}': {len(ds[split_name])} rows")
df = pd.concat(frames, ignore_index=True)
print(f'Total: {len(df)} rows')
print(f'Columns: {list(df.columns)}')

def find_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None
    
MODEL_COL = find_col(['model_name', 'model', 'source_model'])
LABEL_COL = find_col(['llm_annotation', 'label', 'labels', 'behavior', 'annotation'])
PROMPT_COL = find_col(['id', 'trace_id', 'uuid', 'trace_uuid'])

print(f'\nDetected: MODEL={MODEL_COL}, LABEL={LABEL_COL}, PROMPT={PROMPT_COL}')
print(f'Available models: {df[MODEL_COL].unique().tolist()}')

data = []
behaviors = defaultdict(list)
prompt_to_indices = defaultdict(list)
for _, row in df.iterrows():
    if str(row[MODEL_COL]).strip() != DATA_MODEL:
        continue
    if pd.isna(row[LABEL_COL]) or pd.isna(row[PROMPT_COL]):
        continue
        
    d = row.to_dict()
    d[LABEL_COL] = str(d[LABEL_COL]).strip()
    d[PROMPT_COL] = str(d[PROMPT_COL]).strip()
    
    if 'query' not in d:
        for c in ['query', 'prompt', 'question']:
            if c in d and d[c]:
                d['query'] = str(d[c])
                break
                
    if 'sentence' not in d:
        for c in ['sentence', 'target_sentence', 'text']:
            if c in d and d[c]:
                d['sentence'] = str(d[c])
                break
                
    idx = len(data)
    data.append(d)
    behaviors[d[LABEL_COL]].append(idx)
    prompt_to_indices[d[PROMPT_COL]].append(idx)
    
print(f"\nKept {len(data)} entries for '{DATA_MODEL}', {len(behaviors)} behaviors")
if not data:
    print(f"ERROR: no data for '{DATA_MODEL}'")
    print(f'Available: {df[MODEL_COL].unique().tolist()}')
    sys.exit(1)
    
sample = data[0]
assert 'query' in sample and sample['query'], f"Missing 'query' column. Keys: {list(sample.keys())}"
assert 'sentence' in sample and sample['sentence'], f"Missing 'sentence' column. Keys: {list(sample.keys())}"

print(f"  query sample: {str(sample['query'])[:80]}...")
print(f"  sentence sample: {str(sample['sentence'])[:80]}...")

all_prompts = list(prompt_to_indices.keys())
random.shuffle(all_prompts)
split = int(0.8 * len(all_prompts))
train_prompts = set(all_prompts[:split])
test_prompts = set(all_prompts[split:])
train_ids = set()

for pid in train_prompts:
    train_ids.update(prompt_to_indices[pid])
test_ids = set()

for pid in test_prompts:
    test_ids.update(prompt_to_indices[pid])
train, test = ({}, {})
all_ids = list(range(len(data)))

for beh, pos_list in behaviors.items():
    tr_p = [i for i in pos_list if i in train_ids]
    te_p = [i for i in pos_list if i in test_ids]
    tr_n = [i for i in all_ids if i in train_ids and data[i][LABEL_COL] != beh]
    te_n = [i for i in all_ids if i in test_ids and data[i][LABEL_COL] != beh]
    random.shuffle(tr_p)
    random.shuffle(tr_n)
    random.shuffle(te_p)
    random.shuffle(te_n)
    train[beh] = [{'pos': p, 'neg': random.choice(tr_n)} for p in tr_p] if tr_p and tr_n else []
    test[beh] = [{'pos': p, 'neg': random.choice(te_n)} for p in te_p] if te_p and te_n else []
    
with open('data.json', 'w') as f:
    json.dump({'data': data, 'train': train, 'test': test, 'meta': {'dataset': 'ishitakakkar-10/HarmThoughts', 'data_model': DATA_MODEL, 'label_col': LABEL_COL, 'model_col': MODEL_COL, 'prompt_col': PROMPT_COL}}, f)
    
print(f'\nSaved data.json')
print(f'Prompts: train={len(train_prompts)}, test={len(test_prompts)}')
print(f'Sentences: train={len(train_ids)}, test={len(test_ids)}\n')

for b, ids in sorted(behaviors.items(), key=lambda x: -len(x[1])):
    tr = sum((1 for i in ids if i in train_ids))
    te = sum((1 for i in ids if i in test_ids))
    print(f'  {len(ids):>5d} total | {tr:>5d} train | {te:>5d} test | {b}')
