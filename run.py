from main import run_experiment

if __name__ == "__main__":
    run_experiment(model='roberta-base', 
                   task='sst2', 
                   training_type='finetuned',
                   epochs=5, 
                   log_to_wandb=True)
    
    run_experiment(model='roberta-base', 
                   task='sst2', 
                   training_type='frozen',
                   epochs=5, 
                   log_to_wandb=True)
    
    run_experiment(model='roberta-base', 
                   task='sst2', 
                   training_type='optimized',
                   epochs=5, 
                   log_to_wandb=True)
    