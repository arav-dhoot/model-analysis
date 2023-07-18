import yaml
import wandb
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

if task in ['sst2', 'mnli', 'qqp', 'qnli']:
    file_path = 'hparams_yaml_files/large_tasks.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

if task in ['cola', 'rte', 'mrpc', 'stsb']:
    file_path = 'hparams_yaml_files/small_tasks.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

#   TODO: Creating a .yaml file
# TODO: Running the task
# TODO: Setting up wandb logging

# HACK: New naming convention for wandb-logging
#   HACK: Potential group name: [task]
#   HACK: Potential naming convention: [task]-[rand(1e-5 to 1e-6)]

# HACK: Wandb logging
#   HACK: Accuracy, Weight Decay, Num Epochs, Learning Rate
#   HACK: To note => Weight Decay, Num Epochs, Learning Rate, Accuracy

# TODO: Links to consider => https://wandb.ai/amogkam/transformers/reports/Hyperparameter-Optimization-for-Huggingface-Transformers--VmlldzoyMTc2ODI
                             https://huggingface.co/blog/ray-tune
                             https://docs.ray.io/en/latest/tune/examples/pbt_transformers.html#tune-huggingface-example 