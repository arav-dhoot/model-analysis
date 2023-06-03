import torch
from torch.utils.data import Dataset

class QQPDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        question1 = str(self.data[index][0])
        question2 = str(self.data[index][1])
        label = self.data[index][2]

        encoding = self.tokenizer.encode_plus(
            question1,
            question2,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }