import torch
from torch.utils.data import Dataset

class STSBDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']
        similarity_score = item['label']

        encoded_inputs = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'similarity_score': torch.tensor(similarity_score)
        }