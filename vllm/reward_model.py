from vllm.train.trainer import PRM, PRMDataset
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
        text = PRMDataset.format(task, history, action, len(history) - 2)
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

        return scores[0].item()