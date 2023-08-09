
import os
import yaml
import torch
import wandb
from model import Model
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset.mrpcdataset import MRPCDataset

def run():
    file_path = 'yaml_files/mrpc.yaml'
    tokenizer_save_path = 'mrpc_pretrained/tokenizer/'
    model_save_path = 'mrpc_pretrained/model/'

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
    mrpc_dataset = load_dataset('glue', 'mrpc')
    mrpc_num_classes = 3
    mrpc_batch_size = 16

    mrpc_train_data = mrpc_dataset['train']
    mrpc_test_data = mrpc_dataset['validation']
    mrpc_train_dataset = MRPCDataset(mrpc_train_data, tokenizer)
    mrpc_test_dataset = MRPCDataset(mrpc_test_data, tokenizer)
    mrpc_train_dataloader = DataLoader(mrpc_train_dataset, batch_size=mrpc_batch_size, shuffle=True)
    mrpc_test_dataloader = DataLoader(mrpc_test_dataset, batch_size=mrpc_batch_size, shuffle=False)

    model = Model(num_classes=mrpc_num_classes, task='mrpc', training_type='finetune', dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    for epoch in range(1):
        train_loss, train_accuracy, train_loss_list, train_accuracy_list, train_time_list, train_step_list = model.train_epoch(mrpc_train_dataloader, optimizer, device, 2, warmup_ratio=0.06)

    tokenizer.save_pretrained(tokenizer_save_path)
    model.save_pretrained(model_save_path)

if __name__ == '__main__':
    run()