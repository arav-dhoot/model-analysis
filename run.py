from main import run_experiment

if __name__ == "__main__":
    run_experiment(model='roberta-base', 
                   task='qqp', 
                   training_type='finetuned',
                   epochs=5, 
                   log_to_wandb=True, 
                   learning_rate=1e-5)
    
    run_experiment(model='roberta-base', 
                   task='qqp', 
                   training_type='frozen',
                   epochs=5, 
                   log_to_wandb=True, 
                   learning_rate=1e-5)
    
    run_experiment(model='roberta-base', 
                   task='qqp', 
                   training_type='optimized',
                   epochs=5, 
                   log_to_wandb=True, 
                   learning_rate=1e-5)
    