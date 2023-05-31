from transformers import AutoTokenizer
from roberta_model import RoBERTaModel
from datasets import load_dataset
import torch
import wandb
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained('roberta-base')
experiment = 'sst2'

if experiment == 'sst2':
    from dataset.sst2dataset import SST2Dataset

    dataset = load_dataset('glue', 'sst2')

    train_data = dataset['train']
    test_data = dataset['validation']

    train_dataset = SST2Dataset(train_data, tokenizer)
    test_dataset = SST2Dataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=180, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=180, shuffle=False)

if experiment == 'qqp':
    from dataset.qqpdataset import QQPDataset

    dataset = load_dataset("glue", "qqp")

    train_data = dataset['train']
    test_data = dataset['validation']

    train_dataset = SST2Dataset(train_data, tokenizer)
    test_dataset = SST2Dataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create RoBERTa model instance
model = RoBERTaModel(num_classes=2).to(device)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(5):
    train_loss = model.train_epoch(train_dataloader, optimizer, device)
    test_loss, test_accuracy = model.test_epoch(test_dataloader, device)
    
    print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")