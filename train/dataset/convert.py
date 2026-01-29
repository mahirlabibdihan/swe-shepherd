import json
import os

def serialize_history(history):
    text = ""
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|system|>\n{content}\n"
        elif role == "user":
            text += f"<|user|>\n{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>\n{content}\n"
    return text

def extract_prm_steps(example):
    steps = []
    history = []

    for msg in example:
        history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
        if msg["role"] == "assistant":
            steps.append({
                "messages": history.copy(),
                # "response": msg["content"],
                "reward": msg["reward"]
            })
    return steps

prm_train = []

unverified_dataset = []
verified_dataset = []

import glob

unverified_paths = glob.glob(
    "unverified/*.json"
)

for path in unverified_paths:
    with open(path, "r") as f:
        obj = json.load(f)
        unverified_dataset.append(obj)

verified_paths = glob.glob(
    "verified/*.json"
)

for path in verified_paths:
    with open(path, "r") as f:
        obj = json.load(f)
        verified_dataset.append(obj)


for ex in unverified_dataset:
    steps = extract_prm_steps(ex)
    prm_train.extend(steps)

prm_test = []
for ex in verified_dataset:
    steps = extract_prm_steps(ex)
    prm_test.extend(steps)

print(f"Total PRM steps: {len(prm_train)}")

train_rewards = [step["reward"] for step in prm_train]
r_min, r_max = min(train_rewards), max(train_rewards)

def normalize_score(r):
    return (r - r_min) / (r_max - r_min + 1e-8)  # avoid div by zero

for step in prm_train:
    step["reward"] = normalize_score(step["reward"])

for step in prm_test:
    step["reward"] = normalize_score(step["reward"])
    
# split into train and validation
# import random
# random.seed(42)
# random.shuffle(prm_train)
# split_idx = int(0 * len(prm_train))
prm_val = []
# prm_val = prm_train[split_idx:]
# prm_train = prm_train[:split_idx]

# save to prm_train.jsonl, prm_val.jsonl, prm_test.jsonl
def save_jsonl(dataset, path):
    with open(path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
     
import argparse

def main():
    parser = argparse.ArgumentParser(description="Save datasets as JSONL files.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Directory where JSONL files will be saved (default: current directory)"
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    os.makedirs(base_dir, exist_ok=True)  # make sure directory exists

    save_jsonl(prm_train, os.path.join(base_dir, "prm_train.jsonl"))
    save_jsonl(prm_val, os.path.join(base_dir, "prm_val.jsonl"))
    save_jsonl(prm_test, os.path.join(base_dir, "prm_test.jsonl"))
    
if __name__ == "__main__":
    main()