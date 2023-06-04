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
    choices=['sst2', 'qqp', 'wnli', 'cola']
)
parser.add_argument(
    '--epochs',
    help='the number of numbers epochs the model will be trained for',
    type=int
)
parser.add_argument(
    '--log_to_wandb',
    default=True
    type=bool
)
parser.add_argument(
    '--lr',
    default=1e-5,
    type=float
)

arguments = parser.parse_args()

model = arguments.model
task = arguments.task
epochs = arguments.epochs
log_to_wandb = arguments.log_to_wandb
lr = arguments.lr

if __name__ == "__main__":
    run_experiment(model=model, 
                   task=task, 
                   training_type='finetuned',
                   epochs=epochs, 
                   log_to_wandb=log_to_wandb, 
                   learning_rate=lr)
    
    run_experiment(model=model, 
                   task=task, 
                   training_type='frozen',
                   epochs=epochs, 
                   log_to_wandb=log_to_wandb, 
                   learning_rate=lr)
    
    run_experiment(model=model, 
                   task=task, 
                   training_type='optimized',
                   epochs=epochs, 
                   log_to_wandb=log_to_wandb, 
                   learning_rate=lr)
    