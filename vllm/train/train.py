import random
import torch
import json

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn as nn
import torch.nn.functional as F

import json
import glob
from tqdm import tqdm
import os


# read from dataset/prm_train.jsonl

class PRMDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, k=2, max_len=1024):
        self.samples = []
        self.tok = tokenizer
        self.max_len = max_len

        for step in hf_dataset:
            hist = step["history"][-k:]
            task = step["problem_statement"]
            text = self.format(task, hist, step["action"], len(step["history"])-k)
            label = step["reward"]
            
            n_tokens = len(
                tokenizer(
                    text,
                    truncation=False,
                    add_special_tokens=True
                )["input_ids"]
            )
            if n_tokens > max_len:
                continue  # drop sample
            
            self.samples.append((text, label))

    @classmethod
    def format(cls, task, history, action, offset):
        parts = []
        parts.append(f"[Problem Statement]\n{task}")
        # Truncation notice
        if offset > 0:
            parts.append(
                f"[History]\n"
                f"(Earlier {offset} action-observation steps are omitted)\n"
            )
        else:
            parts.append("[History]\n")
        for i, msg in enumerate(history):
            parts.append(f"[Action #{offset+i+1}]\n{msg['action']}")
            parts.append(f"[Observation #{offset+i+1}]\n{msg['observation']}")
        parts.append(f"[Current Action]\n{action}")
        return "\n".join(parts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": label
        }

def collate(batch):
    return {
        "input_ids": pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=0
        ),
        "attention_mask": pad_sequence(
            [b["attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0
        ),
        "labels": torch.tensor([b["label"] for b in batch])
    }

class PRM(nn.Module):
    def __init__(self, model_name, load_lora=True):
        super().__init__()

        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True
        )
        self.model_name = model_name
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=qconf,
            device_map="auto"
        )

        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        
        if load_lora:
            lora = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(base, lora)
        else:
            self.llm = base

        self.value_head = nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        h = out.hidden_states[-1][:, -1]

        self.value_head = self.value_head.to(h.device)

        return self.value_head(h.float()).squeeze(-1)

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        # Save LoRA adapters
        self.llm.save_pretrained(save_dir)

        # Save value head
        torch.save(
            self.value_head.state_dict(),
            os.path.join(save_dir, "value_head.pt")
        )

        # Save metadata (optional but nice)
        meta = {
            "model_name": self.model_name
        }
        torch.save(meta, os.path.join(save_dir, "meta.pt"))
        
    @classmethod
    def load(cls, load_dir: str, device_map="auto"):
        meta = torch.load(os.path.join(load_dir, "meta.pt"))
        model_name = meta["model_name"]

        # IMPORTANT: do NOT attach LoRA here
        model = cls(model_name, load_lora=False)

        # Load LoRA adapters exactly once
        model.llm = PeftModel.from_pretrained(
            model.llm,
            load_dir,
            device_map=device_map
        )

        model.value_head.load_state_dict(
            torch.load(
                os.path.join(load_dir, "value_head.pt"),
                map_location="cpu"
            )
        )

        return model

def main():
    
    dataset = []
    with open("dataset/prm_train.jsonl", "r") as f:
        for line in f:
            obj = json.loads(line)
            dataset.append(obj)
            
    print(dataset[0].keys())

    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    prm_ds = PRMDataset(dataset, tokenizer)
    from torch.utils.data import random_split

    dataset_size = len(prm_ds)
    val_size = int(0.1 * dataset_size)  # 10% for validation
    train_size = dataset_size - val_size

    train_ds, val_ds = random_split(prm_ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate)


    model = PRM(MODEL).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.MSELoss()

    from tqdm import tqdm
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    num_epochs = 1
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # -------- TRAIN --------
        model.train()
        running_train_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for step, batch in enumerate(train_loader_tqdm):
            ids = batch["input_ids"].cuda()
            mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            logits = model(ids, mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            avg_loss_so_far = running_train_loss / (step + 1)

            train_loader_tqdm.set_postfix({"loss": f"{avg_loss_so_far:.4f}"})

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------- VALIDATION --------
        model.eval()
        running_val_loss = 0.0

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        with torch.no_grad():
            for step, batch in enumerate(val_loader_tqdm):
                ids = batch["input_ids"].cuda()
                mask = batch["attention_mask"].cuda()
                labels = batch["labels"].cuda()

                logits = model(ids, mask)
                loss = criterion(logits, labels)

                running_val_loss += loss.item()
                avg_val_so_far = running_val_loss / (step + 1)
                val_loader_tqdm.set_postfix({"val_loss": f"{avg_val_so_far:.4f}"})

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Save the model
    model.save("prm_ckpt")
        
if __name__ == "__main__":
    main()