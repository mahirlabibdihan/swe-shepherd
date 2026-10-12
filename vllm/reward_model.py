from train.trainer import PRM, PRMDataset
from transformers import AutoTokenizer
import torch

class RewardModel:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.model = PRM.load("train/prm_ckpt")
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def process(self, task, history, action):
        act_obs = []
        old_step = {}
        for msg in history:
            if msg["role"] == "assistant":
                old_step['action'] = msg["content"]
            elif msg["role"] == "user":
                old_step['observation'] = msg["content"]
                if 'action' not in old_step:
                    raise ValueError(f"History format error: user message without preceding assistant action.\n{history}")
                act_obs.append(old_step)
                old_step = {}
            
        n_history = 2
        while True:
            text = PRMDataset.format(task, act_obs[-n_history:], action, len(act_obs) - n_history)
            try:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    # truncation=True,
                    # max_length=1024,
                )

                input_ids = inputs["input_ids"].cuda()
                attention_mask = inputs["attention_mask"].cuda()

                with torch.no_grad():
                    scores = self.model(input_ids, attention_mask)
                
                break
            except Exception as e:
                n_history -= 1
                if n_history < 0:
                    raise e
                continue

        return scores[0].item()