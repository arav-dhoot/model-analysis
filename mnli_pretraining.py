import yaml
import torch
from model import Model
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from dataset.mnlidataset import MNLIDataset

def run():
    file_path = 'yaml_files/mnli.yaml'
    tokenizer_save_path = 'mnli_pretrained/tokenizer/'
    model_save_path = 'mnli_pretrained/model/'

    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    weight_decay = data['optimizer']['weight_decay']
    betas = eval(data['optimizer']['adam_betas'])
    eps = float(data['optimizer']['adam_eps'])
    lr = float(data['optimization']['lr'][0])
    dropout = data['model']['dropout']

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    mnli_dataset = load_dataset('glue', 'mnli')
    mnli_num_classes = 3
    mnli_batch_size = 16

    mnli_train_data = mnli_dataset['train']
    mnli_train_dataset = MNLIDataset(mnli_train_data, tokenizer)
    mnli_train_dataloader = DataLoader(mnli_train_dataset, batch_size=mnli_batch_size, shuffle=True)

    model = Model(num_classes=mnli_num_classes, task='mnli', training_type='finetune', dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    for epoch in range(1):
        train_loss, train_accuracy, train_loss_list, train_accuracy_list, train_time_list, train_step_list = model.train_epoch(mnli_train_dataloader, optimizer, device, 2, warmup_ratio=0.06)

    tokenizer.save_pretrained(tokenizer_save_path)
    model.save_pretrained(model_save_path)

if __name__ == '__main__':
    run()