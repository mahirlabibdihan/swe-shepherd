from train import PRMDataset, collate, PRM, criterion
from torch.utils.data import DataLoader
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import torch
from tqdm import tqdm

test_dataset = []
with open("dataset/prm_test.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)
        test_dataset.append(obj)
        
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
test_prm_ds = PRMDataset(test_dataset, tokenizer)
test_loader = DataLoader(test_prm_ds, batch_size=2, shuffle=False, collate_fn=collate)

# Load model from checkpoint
model = PRM(MODEL).cuda()
model.load_state_dict(torch.load("prm_model.pth"))
model.eval()
test_losses = []
with torch.no_grad():
    test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False)
    for step, batch in enumerate(test_loader_tqdm):
        ids = batch["input_ids"].cuda()
        mask = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()

        logits = model(ids, mask)
        loss = criterion(logits, labels)

        test_losses.append(loss.item())

avg_test_loss = sum(test_losses) / len(test_losses)
print(f"Test Loss: {avg_test_loss:.4f}")
