import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer

class RoBERTaModel(nn.Module):
    def __init__(self, num_classes):
        super(RoBERTaModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

    def get_loss(self, logits, labels):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return loss
    
    def train_epoch(self, dataloader, optimizer, device):
        self.train()
        total_loss = 0.0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = self.forward(input_ids, attention_mask)
            loss = self.get_loss(logits, labels)
            loss.backward()
            optimizer.step()
            print(f'Loss is {loss.item()}')
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def test_epoch(self, dataloader, device):
        self.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = self.forward(input_ids, attention_mask)
                loss = self.get_loss(logits, labels)
                
                total_loss += loss.item()
                
                _, predicted_labels = torch.max(logits, dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)
        
        accuracy = correct_predictions / total_predictions
        return total_loss / len(dataloader), accuracy
