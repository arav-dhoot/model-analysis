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

if task == 'sst2': pass
    
if task == 'qqp': pass
    
if task =='wnli': pass

if task == 'cola':
    hparam_file_path = 'hparams_json_files/cola-hparams.json'
    with open(hparam_file_path, 'r') as hparam_file:
        hparam_data = json.load(hparam_file)

if task == 'qnli': pass

if task == 'mnli': pass

if task == 'mrpc':
    hparam_file_path = 'hparams_json_files/mrpc-hparams.json'
    with open(hparam_file_path, 'r') as hparam_file:
        hparam_data = json.load(hparam_file)

if task == 'rte':
    hparam_file_path = 'hparams_json_files/rte-hparams.json'
    with open(hparam_file_path, 'r') as hparam_file:
        hparam_data = json.load(hparam_file)

if task == 'stsb': pass

if task == 'stsb': num_classes = 1
if task == 'mnli': num_classes = 3
else: num_classes = 2

batch_size = hparam_data['batch_size']
max_tokens = 512
weight_decay = 0.1
betas = (0.9,0.98)
eps = 1e-06
lr = hparam_data['learning_rate']
epochs = hparam_data['epochs']
dropout = 0.1
warmup_ratio = hparam_data['warmup_ratio']

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