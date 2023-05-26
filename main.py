from sst2dataset import SST2Dataset
from transformers import RobertaTokenizer
from roberta_model import RoBERTaModel
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Load the SST-2 dataset
dataset = load_dataset('glue', 'sst2')

train_data = dataset['train']
test_data = dataset['test']

train_dataset = SST2Dataset(train_data, tokenizer)
test_dataset = SST2Dataset(test_data, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create RoBERTa model instance
model = RoBERTaModel(num_classes=2).to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(5):
    train_loss = model.train_epoch(train_dataloader, optimizer, device)
    import pdb; pdb.set_trace()
    test_loss, test_accuracy = model.test_epoch(test_dataloader, device)
    
    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")