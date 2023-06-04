import json
import tqdm
import time
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModel

class Model(nn.Module):
    def __init__(self, 
                 num_classes,
                 task,
                 training_type,
                 model='roberta-base', 
                ):
        
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(model)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.model.config.hidden_size, num_classes)
            )
        self.name_list = [name for name, params in self.model.named_parameters()]
        self.grad_dict = dict()
        for name in self.name_list: self.grad_dict[name] = list()
        self.training_type = training_type
        self.task = task
        self.trainable_params = 0
        for name, param in self.model.named_parameters(): self.trainable_params += torch.numel(param)
        self.trained_parameters = 0
        self.trained_proportion = 0 

        if self.training_type == 'finetune':
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            self.trained_proportion = 1
        
        elif self.training_type == 'frozen':
            for name, param in self.model.named_parameters():
                if 'embeddings' in name or 'pooler' in name:
                    param.requires_grad = True
                    self.trained_parameters += torch.numel(param)
                else: param.requires_grad = False
            self.trained_proportion = self.trained_parameters/self.trainable_params

        elif self.training_type == 'optimized':
            path = task+'-data.json' 
            with open(path) as file: self.grad_dict = json.load(file)
            var_dict = dict()
            for key in self.grad_dict.keys(): var_dict[key] = torch.var(torch.tensor(self.grad_dict[key]))
            sorted_var_dict = dict(sorted(var_dict.items(), key=lambda x: x[1]))
            key_list, value_list = [], []
            for key, value in sorted_var_dict.items():
                value_list.append(value) 
                key_list.append(key)
            value_list = np.array(np.delete(value_list,0))
            counter=0
            cumulative_list = np.array(np.cumsum(value_list)/np.sum(value_list))
            for value in cumulative_list:
                if value > 0.01: break
                else: counter+=1
            value=counter-1

            for name, param in self.model.named_parameters():
                if 'embeddings' in name or 'pooler' in name:
                    param.requires_grad = True
                    self.trained_parameters += torch.numel(param)
                for key in key_list[value:]: 
                    if name in key and ('embeddings' not in name or 'pooler' not in name):
                        param.requires_grad = True
                        self.trained_parameters += torch.numel(param)
                    else: param.requires_grad = False
            self.trained_proportion = self.trained_parameters/self.trainable_params
        
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
        total_correct = 0
        batch_count = 0 
        start_time = time.time() 
        loss_list = list()
        accuracy_list = list()
        
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if self.task != 'sst2':
                labels = batch['label'].to(device) 
            else: labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = self.forward(input_ids, attention_mask)
            loss = self.get_loss(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loss_list.append(loss.item())
            predicted_labels = torch.argmax(logits, dim=1)
            total_correct += (predicted_labels == labels).sum().item()  
            accuracy_list.append((predicted_labels == labels).sum().item()/len(labels))
            
            counter = 0
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                elif param.grad is not None:
                    self.grad_dict[name].append(round(torch.norm(param.grad).item(), 3))
            counter += 1 

            batch_count += 1
            
        accuracy = total_correct / len(dataloader.dataset)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return total_loss / len(dataloader), accuracy, elapsed_time/batch_count, loss_list, accuracy_list
    
    def test_epoch(self, 
                   dataloader, 
                   device
                   ):
        
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        batch_count = 0
        start_time = time.time()
        loss_list = list()
        accuracy_list = list()
        
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                if self.task != 'sst2':
                    labels = batch['label'].to(device) 
                else: labels = batch['labels'].to(device)
                logits = self.forward(input_ids, attention_mask)
                loss = self.get_loss(logits, labels)
                
                total_loss += loss.item()
                loss_list.append(loss.item())
                _, predicted_labels = torch.max(logits, dim=1)
                total_correct += (predicted_labels == labels).sum().item()
                accuracy_list.append((predicted_labels == labels).sum().item()/len(labels))
                total_predictions += labels.size(0)

                batch_count += 1
                
        accuracy = total_correct / total_predictions
        end_time = time.time()
        elapsed_time = end_time - start_time

        return total_loss / len(dataloader), accuracy, elapsed_time/batch_count, loss_list, accuracy_list

    def file_write(self):
        file_name = f'{self.task}-data.json'
        try:
            file = open(file_name, 'x')
            with open(file_name, 'w') as file:
                json.dump(self.grad_dict, file, indent=4)
        except:
            with open(file_name, 'w') as file:
                json.dump(self.grad_dict, file, indent=4)
        return file_name

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