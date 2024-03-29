import torch
from torch.utils.data import Dataset

class MRPCDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence1 = self.data[index]['sentence1']
        sentence2 = self.data[index]['sentence2']
        label = self.data[index]['label']
        
        encoding = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)
        }
