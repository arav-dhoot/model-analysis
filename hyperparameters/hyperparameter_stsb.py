import yaml
import json
import optuna
from main import run_experiment


def objective(trial):
    file_path = './hparams_yaml_files/small_tasks.yaml'
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    epochs = trial.suggest_categorical('epochs', data['max_epochs'])
    learning_rate = trial.suggest_categorical('learning_rate', [float(value) for value in data['learning_rate']])
    batch_size = trial.suggest_categorical('batch_size', data['batch_size'])
    warmup_ratio = trial.suggest_categorical('warmup_ratio', data['warm_up'])

    file_path = './yaml_files/sts_b.yaml'
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
        task='stsb',
        training_type='finetuned',
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
        warmup_ratio=warmup_ratio,
        project_name='model_hyperparameter_search',
        sweep=True
    )

    return accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize") 
    study.optimize(objective, n_trials=20) 
    print("Best trial:")
    trial = study.best_trial
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    file_name = 'stsb-hparams.json'
    try:
        with open(file_name, 'w') as file:
            json.dump(trial.params, file, indent=4)
    except:
        with open(file_name, 'w') as file:
            json.dump(trial.params, file, indent=4)