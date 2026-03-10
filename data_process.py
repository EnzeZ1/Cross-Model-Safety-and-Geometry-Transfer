import json
import random
from collections import Counter

"""
data_process.py

This script prepares the dataset used for steering vector extraction in our project.
It converts raw behavior-labeled reasoning traces into a structured format suitable
for computing steering vectors.

The script performs the following steps:
1. Load raw sentence-level reasoning data.
2. Normalize model names and behavior labels.
3. Construct positive/negative sentence sets for each safety behavior.
4. Generate paired training and testing samples.
5. Export a unified data.json file for downstream modules
   (get_steering.py, DoM.py, and eval.py).

The preprocessing design is inspired by the paper
"Annotating the Chain-of-Thought: A Behavior-Labeled Dataset for AI Safety",
which introduces sentence-level safety behavior annotations for reasoning traces.
Following the paper, we treat sentences with a given behavior label as positive
samples and sentences without that label as negative samples, enabling
Difference-of-Means (DoM) based steering vector computation.

Author: Yuliang Wu
"""


SEED = 42
random.seed(SEED)

# Canonical behavior names used in the project
BEHAVIORS = [
    "rephrase_prompt",
    "speculate_user_motive",
    "flag_user_testing",
    "flag_prompt_as_harmful",
    "state_safety_concern",
    "state_legal_concern",
    "state_ethical_moral_concern",
    "express_uncertainty_confusion",
    "self_correct_info_detail",
    "state_fact_knowledge",
    "plan_immediate_reasoning_step",
    "summarize_internal_reasoning",
    "intend_refusal_or_safe_action",
    "consider_benign_reinterpretation",
    "suggest_safe_constructive_alternative",
    "intend_harmful_compliance",
    "detail_harmful_method_or_info",
    "note_risk_while_detailing_harm",
    "neutral_filler_transition",
    "other"
]

# Normalize model names so they match get_steering.py
MODEL_MAP = {
    "Qwen3-8B": "qwen3-8b",
    "qwen3-8b": "qwen3-8b",

    "DeepSeek-R1-8B": "deepseek-r1-8b",
    "deepseek-r1-8b": "deepseek-r1-8b",

    "DeepSeek-R1-8B-0528": "deepseek-r1-8b-0528",
    "DeepSeek-R1-0528-Qwen3-8B": "deepseek-r1-8b-0528",
    "deepseek-r1-8b-0528": "deepseek-r1-8b-0528",

    "DeepSeek-R1-32B": "deepseek-r1-32b",
    "DeepSeek-R1-Distill-Qwen-32B": "deepseek-r1-32b",
    "deepseek-r1-32b": "deepseek-r1-32b",

    "DeepSeek-R1-Llama-8B": "deepseek-r1-llama-8b",
    "DeepSeek-R1-Distill-Llama-8B": "deepseek-r1-llama-8b",
    "deepseek-r1-llama-8b": "deepseek-r1-llama-8b",
}

# Optional label alias mapping
LABEL_MAP = {
    "intend_refusal": "intend_refusal_or_safe_action",
    "intend_safe_action": "intend_refusal_or_safe_action",
    "state_ethical_concern": "state_ethical_moral_concern",
    "state_moral_concern": "state_ethical_moral_concern",
    "detail_harmful_method_info": "detail_harmful_method_or_info",
    "note_risk_while_detailing": "note_risk_while_detailing_harm",
    "neutral_filler": "neutral_filler_transition",
    "state_fact": "state_fact_knowledge",
    "summarize_reasoning": "summarize_internal_reasoning",
    "plan_reasoning_step": "plan_immediate_reasoning_step",
    "self_correct": "self_correct_info_detail",
    "express_uncertainty": "express_uncertainty_confusion",
    "suggest_safe_alternative": "suggest_safe_constructive_alternative",
}

ALLOWED_MODELS = {
    "qwen3-8b",
    "deepseek-r1-8b",
    "deepseek-r1-8b-0528",
    "deepseek-r1-32b",
    "deepseek-r1-llama-8b",
}


def clean_text(x):
    """Convert any value into a clean one-line string."""
    if x is None:
        return ""
    return " ".join(str(x).strip().split())


def normalize_key(x):
    """Normalize a label-like string into lowercase snake_case."""
    s = clean_text(x).lower()
    for ch in ["/", "-", ",", ";", ":", ".", "(", ")", "[", "]"]:
        s = s.replace(ch, " ")
    return "_".join(s.split())


def normalize_model(model_name):
    """Map raw model names to canonical project model names."""
    model_name = clean_text(model_name)
    if model_name in MODEL_MAP:
        return MODEL_MAP[model_name]
    return normalize_key(model_name).replace("_", "-")


def normalize_labels(labels):
    """Convert raw labels into a canonical label list."""
    if labels is None:
        return []

    if isinstance(labels, str):
        if labels.startswith("[") and labels.endswith("]"):
            try:
                labels = json.loads(labels)
            except:
                labels = [labels]
        elif "|" in labels:
            labels = [x.strip() for x in labels.split("|") if x.strip()]
        elif "," in labels:
            labels = [x.strip() for x in labels.split(",") if x.strip()]
        else:
            labels = [labels]

    if not isinstance(labels, list):
        labels = [labels]

    out = []
    for lb in labels:
        lb = normalize_key(lb)
        lb = LABEL_MAP.get(lb, lb)
        if lb in BEHAVIORS:
            out.append(lb)

    return sorted(set(out))


def load_data(path):
    """Load raw data from .json or .jsonl."""
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "examples", "records", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]

    raise ValueError("Unsupported input format.")


def pick(raw, keys):
    """Pick the first non-empty field from a list of possible keys."""
    for k in keys:
        if k in raw and raw[k] not in [None, "", [], {}]:
            return raw[k]
    return None


def convert_example(raw):
    """
    Convert one raw example into the format required by get_steering.py.

    Output fields:
        prompt
        context
        target_sentence
        model
        labels
    """
    prompt = clean_text(pick(raw, ["prompt", "user_prompt", "instruction", "input", "query"]))
    context = clean_text(pick(raw, ["context", "history", "prefix", "reasoning_prefix"]) or "")
    target_sentence = clean_text(pick(raw, ["target_sentence", "sentence", "text", "target", "content"]))
    labels = normalize_labels(pick(raw, ["labels", "behaviors", "behavior_labels", "tags", "annotations"]))
    model = normalize_model(pick(raw, ["model", "model_name", "source_model", "generator"]))

    if not prompt or not target_sentence or not model:
        return None

    if model not in ALLOWED_MODELS:
        return None

    return {
        "prompt": prompt,
        "context": context,
        "target_sentence": target_sentence,
        "model": model,
        "labels": labels
    }


def split_ids(ids, test_ratio=0.2):
    """Split indices into train and test."""
    ids = ids[:]
    random.shuffle(ids)

    if len(ids) < 5:
        n_test = max(0, len(ids) // 5)
    else:
        n_test = max(1, int(len(ids) * test_ratio))

    return ids[n_test:], ids[:n_test]


def build_pairs(pos_ids, neg_ids):
    """Build one-to-one positive-negative pairs."""
    pos_ids = pos_ids[:]
    neg_ids = neg_ids[:]
    random.shuffle(pos_ids)
    random.shuffle(neg_ids)

    n = min(len(pos_ids), len(neg_ids))
    return [{"pos": pos_ids[i], "neg": neg_ids[i]} for i in range(n)]


def build_dataset(data, test_ratio=0.2):
    """
    Build train/test pairs for each behavior.
    Pairing is done within each model first to match downstream code.
    """
    train = {}
    test = {}

    models = sorted(set(x["model"] for x in data))

    for behavior in BEHAVIORS:
        train_pairs = []
        test_pairs = []

        for model in models:
            model_ids = [i for i, x in enumerate(data) if x["model"] == model]
            pos_ids = [i for i in model_ids if behavior in data[i]["labels"]]
            neg_ids = [i for i in model_ids if behavior not in data[i]["labels"]]

            if len(pos_ids) < 2 or len(neg_ids) < 2:
                continue

            pos_train, pos_test = split_ids(pos_ids, test_ratio)
            neg_train, neg_test = split_ids(neg_ids, test_ratio)

            train_pairs.extend(build_pairs(pos_train, neg_train))
            test_pairs.extend(build_pairs(pos_test, neg_test))

        if train_pairs:
            train[behavior] = train_pairs
        if test_pairs:
            test[behavior] = test_pairs

    return train, test


def print_stats(data, train, test):
    """Print simple dataset statistics."""
    print(f"Total valid samples: {len(data)}")
    print("Model distribution:", dict(Counter(x["model"] for x in data)))
    print(f"Number of train tasks: {len(train)}")
    print(f"Number of test tasks: {len(test)}")
    print("-" * 60)

    for behavior in BEHAVIORS:
        pos = sum(1 for x in data if behavior in x["labels"])
        neg = len(data) - pos
        tr = len(train.get(behavior, []))
        te = len(test.get(behavior, []))
        if tr > 0 or te > 0:
            print(f"{behavior}: pos={pos}, neg={neg}, train_pairs={tr}, test_pairs={te}")


def main():
    # Change these paths if needed
    input_path = "raw_data.json"
    output_path = "data.json"

    raw_data = load_data(input_path)

    data = []
    for raw in raw_data:
        item = convert_example(raw)
        if item is not None:
            data.append(item)

    if not data:
        raise ValueError("No valid examples found. Please check raw field names in convert_example().")

    train, test = build_dataset(data, test_ratio=0.2)

    output = {
        "data": data,
        "train": train,
        "test": test
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print_stats(data, train, test)
    print(f"Saved processed data to {output_path}")


if __name__ == "__main__":
    main()