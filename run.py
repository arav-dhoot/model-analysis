import json
import yaml
import argparse
from main import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    help='the model type',
    default='roberta-base'
)
parser.add_argument(
    '--task',
    help='one of the GLUE tasks',
    choices=['sst2', 'qqp', 'wnli', 'cola', 'qnli', 'mnli', 'mrpc', 'rte', 'stsb']
)

parser.add_argument(
    '--log_to_wandb',
    default=True
)

arguments = parser.parse_args()

model = arguments.model
task = arguments.task
log_to_wandb = arguments.log_to_wandb

if task == 'sst2':
    file_path = 'yaml_files/sst_2.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
if task == 'qqp':
    file_path = 'yaml_files/qqp.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
if task =='wnli':
    file_path = 'yaml_files/rte.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

if task == 'cola':
    file_path = 'yaml_files/cola.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

if task == 'qnli':
    file_path = 'yaml_files/qnli.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

if task == 'mnli':
    file_path = 'yaml_files/mnli.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

if task == 'mrpc':
    file_path = 'yaml_files/mrpc.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

if task == 'rte':
    file_path = 'yaml_files/rte.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

if task == 'stsb':
    file_path = 'yaml_files/sts_b.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

num_classes = data['task']['num_classes']
batch_size = data['dataset']['batch_size']
weight_decay = data['optimizer']['weight_decay']
betas = eval(data['optimizer']['adam_betas'])
eps = float(data['optimizer']['adam_eps'])
lr = float(data['optimization']['lr'][0])
epochs = data['optimization']['max_epoch']
dropout = data['model']['dropout']

max_tokens = 4400
warmup_ratio = 0.1

if __name__ == "__main__":
    run_experiment(model=model, 
                   task=task, 
                   training_type='finetuned',
                   epochs=epochs, 
                   log_to_wandb=log_to_wandb, 
                   learning_rate=lr,
                   num_classes=num_classes,
                   batch_size=batch_size, 
                   dropout=dropout,
                   max_tokens=max_tokens, 
                   weight_decay=weight_decay,
                   betas=betas,
                   eps=eps, 
                   warmup_ratio=warmup_ratio)
    
    run_experiment(model=model, 
                   task=task, 
                   training_type='frozen',
                   epochs=epochs * 2, 
                   log_to_wandb=log_to_wandb, 
                   learning_rate=lr,
                   num_classes=num_classes,
                   batch_size=batch_size, 
                   dropout=dropout,
                   max_tokens=max_tokens, 
                   weight_decay=weight_decay,
                   betas=betas,
                   eps=eps,
                   warmup_ratio=warmup_ratio)
    
    run_experiment(model=model, 
                   task=task, 
                   training_type='optimized',
                   epochs=epochs * 2, 
                   log_to_wandb=log_to_wandb, 
                   learning_rate=lr,
                   num_classes=num_classes,
                   batch_size=batch_size, 
                   dropout=dropout,
                   max_tokens=max_tokens, 
                   weight_decay=weight_decay,
                   betas=betas,
                   eps=eps,
                   warmup_ratio=warmup_ratio)