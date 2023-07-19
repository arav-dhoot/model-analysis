import yaml
import optuna
from main import run_experiment

def objective(trial):
    file_path = './hparams_yaml_files/small_tasks.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    epochs = trial.suggest_categorical('epochs', data['max_epochs'])
    learning_rate = trial.suggest_categorical('learning_rate', float(data['learning_rate']))
    batch_size = trial.suggest_categorical('batch_size', data['batch_size'])

    file_path = './yaml_files/mrpc.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    num_classes = data['task']['num_classes']

    max_tokens = 512
    eps = 1e-6
    betas = (0.9, 0.98)
    dropout = 0.1
    weight_decay = 0.01 

    accuracy = run_experiment(
        model='roberta-base',
        task='mrpc',
        training_type='finetune',
        epochs=epochs,
        log_to_wandb=True,
        learning_rate=learning_rate,
        num_classes=num_classes,
        batch_size=batch_size,
        dropout=dropout,
        max_tokens=max_tokens,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps, 
        lr_scheduler='linear',
        scheduler_updates=None,
        project_name='model_hyperparameter_search'
    )

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize") 
    study.optimize(objective, n_trials=100) 
    print("Best trial:")
    trial = study.best_trial
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")