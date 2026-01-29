import json

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

    for msg in example["messages"]:
        if msg["role"] == "assistant":
            steps.append({
                "messages": history.copy(),
                "response": msg["content"],
                "reward": msg["reward"]
            })

        history.append({
            "role": msg["role"],
            "content": msg["content"]
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
print(prm_train[0])


train_rewards = [step["reward"] for step in prm_train]
r_min, r_max = min(train_rewards), max(train_rewards)

def normalize_score(r):
    return (r - r_min) / (r_max - r_min + 1e-8)  # avoid div by zero

for step in prm_train:
    step["reward"] = normalize_score(step["reward"])

for step in prm_test:
    step["reward"] = normalize_score(step["reward"])
    
# split into train and validation
import random
random.seed(42)
random.shuffle(prm_train)
split_idx = int(0.9 * len(prm_train))
prm_val = prm_train[split_idx:]
prm_train = prm_train[:split_idx]

# save to prm_train.jsonl, prm_val.jsonl, prm_test.jsonl
def save_jsonl(dataset, path):
    with open(path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
save_jsonl(prm_train, "prm_train.jsonl")
save_jsonl(prm_val, "prm_val.jsonl")
save_jsonl(prm_test, "prm_test.jsonl")