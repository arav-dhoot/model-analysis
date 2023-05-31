import json
import torch
import torch.nn as nn
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, 
                 num_classes,
                 model='roberta-base', 
                 training_type='finetune',
                 ):
        
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(model)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.model.config.hidden_size, num_classes)
        self.name_list = [name for name, params in self.model.named_parameters()]
        self.grad_dict = dict()
        for name in self.name_list:
            self.grad_dict[name] = list()
        
    def forward(self, 
                input_ids, 
                attention_mask
                ):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

    def get_loss(self, 
                 logits, 
                 labels
                ):
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return loss
    
    def train_epoch(self, 
                    dataloader, 
                    optimizer, 
                    device
                    ):
        
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
    
    def test_epoch(self, 
                   dataloader, 
                   device
                   ):
        
        self.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                import pdb; pdb.set_trace()
                logits = self.forward(input_ids, attention_mask)
                loss = self.get_loss(logits, labels)
                
                total_loss += loss.item()
                print(f'Loss is {loss.item()}')
                _, predicted_labels = torch.max(logits, dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                total_predictions += labels.size(0)
        
        accuracy = correct_predictions / total_predictions
        return total_loss / len(dataloader), accuracy

    def file_write(self):
        file_name = f'{self.task}-data.json'
        try:
            file = open(file_name, 'x')
            with open(file_name, 'w') as file:
                json.dump(self.grad_dict, file, indent=4)
        except:
            with open(file_name, 'w') as file:
                json.dump(self.grad_dict, file, indent=4)

    def calculate_stats(self, 
                        var=True, 
                        mean=False, 
                        top_n=5
                        ):
        
        if var:
            var_dict = dict()
            var_list = list()
            for key in self.grad_dict.keys():
                var_dict[key] = torch.var(torch.tensor(self.grad_dict[key])) 
            var_dict = dict(sorted(var_dict.items(), key=lambda item: item[1]))
            for item in list(reversed(var_dict.keys()))[:top_n]:
                print(f'{item} => {var_dict[item]}')
                var_list.append(var_dict[item])
                if not mean: return var_list
        if mean:
            mean_dict = dict()
            mean_list = list()
            for key in self.grad_dict.keys():
                mean_dict[key] = torch.mean(torch.tensor(self.grad_dict[key])) 
            mean_dict = dict(sorted(mean_dict.items(), key=lambda item: item[1]))
            for item in list(reversed(mean_dict.keys()))[:top_n]:
                print(f'{item} => {mean_dict[item]}')
                mean_list.append(mean_dict[item])
                if not var: return mean_list
        return var_list, mean_list        