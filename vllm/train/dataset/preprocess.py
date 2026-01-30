import json
import glob
from tqdm import tqdm

dataset = []

paths = glob.glob(
    "annotated/*.annotated.json"
)

for path in tqdm(paths, desc="Loading JSON files"):
    with open(path, "r") as f:
        obj = json.load(f)
        dataset.append({
            "file_path": path,
            **obj
        })
        


print(dataset[0].keys())

results = {}

# read results.json
with open("results.json", "r") as f:
    results = json.load(f)

reward_scores = {
    "COMMAND_SUCCESS": 0.0,
    "READ_TARGET_FILE": 0.15,
    "READ_TEST_FILE": 0.05,
    "CREATE_TEST": 0.1,
    "EDIT_TARGET_FILE": 0.15,
    "RUN_TEST_AFTER_EDIT": 0.15,
    "F2P_TEST_PASS": 0.3,
    # "P2P_TEST_MAINTAINED": 0.0,  # optional, remove
    "READ_RELATED_FILE": 0.05
}

penalty_scores = {
    "COMMAND_ERROR": -0.3,
    "REPETITION_EXACT": -0.05,
    "P2P_TEST_FAIL": -0.3,
    # "TIMEOUT": -0.1
}

def calculate_reward(rewards, penalties):
    total = 0.0
    for r in rewards:
        total += reward_scores.get(r, 0.0)
    for p in penalties:
        total += penalty_scores.get(p, 0.0)
    return total

# Calculate reward
def add_rewards(example):
    new_msgs = []
    try:
        flag = False
        for i, msg in enumerate(example["messages"]):
            if msg.get("role") == "assistant":
                msg = dict(msg)
                if "COMMAND_ERROR" in msg["penalties"] and "COMMAND_SUCCESS" in msg["rewards"]:
                    msg["penalties"].remove("COMMAND_ERROR") 
                msg["reward"] = calculate_reward(msg["rewards"], msg["penalties"])  
                if i == len(example["messages"]) - 2:
                    # Terminating action : orm
                    if example["instance_id"] in results["resolved"]:
                        msg["reward"] = 1.0
                    flag = True
            new_msgs.append(msg)
        example["messages"] = new_msgs
        
    except Exception as e:
        print(f"Error processing example {example.keys()}: {e}")
    return example

dataset = [add_rewards(ex) for ex in dataset]

def discounted_reward(example, gamma=0.6):
    messages = example["messages"]
    G = 0.0
    new_msgs = []
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            G = msg.get("reward", 0.0) + gamma * G
            msg = dict(msg)
            msg["reward"] = G
            del msg["rewards"]
            del msg["penalties"]
            del msg["reward_metadata"]
            del msg["extra"]
        new_msgs.insert(0, msg)
    example["messages"] = new_msgs
    return example

dataset = [discounted_reward(ex) for ex in dataset]

from datasets import load_dataset
ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")
verified_instance_ids = set(ds['instance_id'])
verified_dataset = [ex for ex in dataset if ex['instance_id'] in verified_instance_ids]
unverified_dataset = [ex for ex in dataset if ex['instance_id'] not in verified_instance_ids]

# create unverified directory
import os
os.makedirs("unverified", exist_ok=True)
for ex in unverified_dataset:
    base_name = os.path.basename(ex["file_path"])
    # remove .annotated from the file name
    base_name = base_name.replace(".annotated", "")
    with open(os.path.join("unverified", base_name), "w") as f:
        json.dump(ex["messages"], f, indent=2)
        
os.makedirs("verified", exist_ok=True)
for ex in verified_dataset:
    base_name = os.path.basename(ex["file_path"])
    # remove .annotated from the file name
    base_name = base_name.replace(".annotated", "")
    with open(os.path.join("verified", base_name), "w") as f:
        json.dump(ex["messages"], f, indent=2)

