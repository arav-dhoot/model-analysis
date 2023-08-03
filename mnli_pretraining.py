
import yaml
import torch
import wandb
from model import Model
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset.mnlidataset import MNLIDataset

def run():
    file_path = 'yaml_files/mnli.yaml'
    save_path = 'mnli_pretrained/'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    num_classes = data['task']['num_classes']
    batch_size = data['dataset']['batch_size']
    max_tokens = data['dataset']['max_tokens']
    weight_decay = data['optimizer']['weight_decay']
    betas = eval(data['optimizer']['adam_betas'])
    eps = float(data['optimizer']['adam_eps'])
    lr = float(data['optimization']['lr'][0])
    epochs = data['optimization']['max_epoch']
    dropout = data['model']['dropout']

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    mnli_dataset = load_dataset('glue', 'mnli')
    mnli_num_classes = 3
    mnli_batch_size = 16

    mnli_train_data = mnli_dataset['train']
    mnli_test_data = mnli_dataset['validation_matched']
    mnli_train_dataset = MNLIDataset(mnli_train_data, tokenizer)
    mnli_test_dataset = MNLIDataset(mnli_test_data, tokenizer)
    mnli_train_dataloader = DataLoader(mnli_train_dataset, batch_size=mnli_batch_size, shuffle=True)
    mnli_test_dataloader = DataLoader(mnli_test_dataset, batch_size=mnli_batch_size, shuffle=False)

    model = Model(num_classes=mnli_num_classes, task='mnli', training_type='finetune', dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    for epoch in range(1):
        train_loss, train_accuracy, train_loss_list, train_accuracy_list, train_time_list, train_step_list = model.train_epoch(mnli_train_dataloader, optimizer, device, 2, warmup_ratio=0.06)
        test_loss, test_accuracy, test_loss_list, test_accuracy_list, test_time_list, test_step_list = model.test_epoch(mnli_test_dataloader, device)

        for tr_loss, tr_accuracy, tr_time, tr_step in zip(train_loss_list, train_accuracy_list, train_time_list, train_step_list):
            wandb.log(
                {
                    'Train Loss':tr_loss,
                    'Train Accuracy':tr_accuracy,
                    'Train Time':tr_time,
                    'Train Step': (epoch * len(train_step_list)) + tr_step
                }, 
            )
        
        for te_loss, te_accuracy, te_time, te_step in zip(test_loss_list, test_accuracy_list, test_time_list, test_step_list):
            wandb.log(
                {
                    'Test Loss (batch)':te_loss,
                    'Test Accuracy (batch)':te_accuracy,
                    'Test Time':te_time,
                    'Test Step': (epoch * len(test_step_list)) + te_step 
                },   
            )

        print(f'Epoch {epoch + 1} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}')

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)

if __name__ == '__main__':
    run()