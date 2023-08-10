import json
import tqdm
import time
import torch
import wandb
import numpy as np
import torch.nn as nn
from scipy.stats import pearsonr
from transformers import AutoModel
from torch.optim.lr_scheduler import LambdaLR

class Model(nn.Module):
    def __init__(self, 
                 num_classes,
                 task,
                 training_type,
                 dropout,
                 model='roberta-base', 
                ):
        
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size),
            nn.GELU(),
            self.dropout,
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
            counter=0
            cumulative_list = np.array(np.cumsum(value_list)/np.sum(value_list))
            for value in cumulative_list:
                if value > 0.01: break
                else: counter+=1
            value=counter-1

            for name, param in self.model.named_parameters(): 
                    if name in key_list[value:] or ('embeddings' in name or 'pooler' in name):
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
        logits = self.fc(pooled_output)
        return logits

    def get_loss(self, 
                 logits, 
                 labels
                ):
        
        if self.task == 'stsb':
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        return loss
    
    def train_epoch(self, 
                    dataloader, 
                    optimizer, 
                    device,
                    epochs,
                    warmup_ratio = 0.06
                    ):
        
        self.train()
        total_loss = 0.0
        total_correct = 0
        batch_count = 0 
        batch_list = list()
        time_list = list()
        loss_list = list()
        accuracy_list = list()

        def lr_lambda(batch):
            if batch < (warmup_ratio * len(dataloader) * epochs):
                return float(batch) / float(max(1, (warmup_ratio * len(dataloader) * epochs)))
            return max(0.0, float(len(dataloader) - batch) / float(max(1, len(dataloader) -  (warmup_ratio * len(dataloader) * epochs))))
  
        # TODO: wandb.log(learning_rate)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        for batch in tqdm.tqdm(dataloader):
            start_time = time.time()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                logits = self.forward(input_ids, attention_mask)
                loss = self.get_loss(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            loss_list.append(loss.item())
            predicted_labels = torch.argmax(logits, dim=1)

            predicted_labels = predicted_labels.to(torch.int32)
            labels = labels.to(torch.int32)

            total_correct += (predicted_labels == labels).sum().item()  
            accuracy_list.append((predicted_labels == labels).sum().item()/len(labels))
            
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                elif param.grad is not None:
                    self.grad_dict[name].append(round(torch.norm(param.grad).item(), 3))
            batch_count += 1
            batch_list.append(batch_count)
            end_time = time.time()
            time_list.append(end_time - start_time)

        accuracy = total_correct / len(dataloader.dataset)
        
        return total_loss / len(dataloader), accuracy, loss_list, accuracy_list, time_list, batch_list
    
    def test_epoch(self, 
                   dataloader, 
                   device
                   ):
        
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_predictions = 0
        batch_count = 0
        batch_list = list()
        time_list = list()
        loss_list = list()
        accuracy_list = list()
        predicted = list()
        actual_labels = list()
        
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                start_time = time.time()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                    logits = self.forward(input_ids, attention_mask)
                    loss = self.get_loss(logits, labels)
                
                total_loss += loss.item()
                loss_list.append(loss.item())
                _, predicted_labels = torch.max(logits, dim=1)

                predicted.extend(predicted_labels.cpu().numpy().tolist())
                actual_labels.extend(labels.cpu().numpy().tolist())

                total_correct += (predicted_labels == labels).sum().item()
                accuracy_list.append((predicted_labels == labels).sum().item()/len(labels))
                total_predictions += labels.size(0)
                batch_count += 1
                batch_list.append(batch_count)
                end_time = time.time()
                time_list.append(end_time - start_time)
        if self.task == 'stsb': accuracy = pearsonr(predicted, actual_labels)
        else: accuracy = total_correct / total_predictions

        return total_loss / len(dataloader), accuracy, loss_list, accuracy_list, time_list, batch_list

    def save_pretrained(self, file_path):
        self.model.save_pretrained(file_path)

    def from_pretrained(self, file_path):
        pass
        # TODO: implementation
    
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