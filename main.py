import time
import torch
import wandb
import random
from model import Model
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def run_experiment (
        model,
        task,
        training_type,
        epochs, 
        log_to_wandb,
        learning_rate
):

    tokenizer = AutoTokenizer.from_pretrained(model)
    task = task
    training_type = training_type
    log_to_wandb = log_to_wandb
    epochs = epochs
    learning_rate = learning_rate

    short_name = str(random.randint(int(1e5), int(1e6) - 1))
    run_name = f'{task}-{training_type}-{short_name}'

    if log_to_wandb:
        config = dict(
            short_name=short_name,
            run_name=run_name,
            task=task,
            training_type=training_type,
            epochs=epochs,
            learning_rate=learning_rate
        )

        wandb.init(
            project='model_analysis',
            group=f'{time.strftime("%m-%d", time.localtime(time.time()))}-{task}',
            name=f'{task}-{training_type}-{short_name}',
            config=config,
        )

    if task == 'sst2':
        from dataset.sst2dataset import SST2Dataset

        dataset = load_dataset('glue', 'sst2')
        num_classes = 2
        batch_size = 400

        train_data = dataset['train']
        test_data = dataset['validation']
        train_dataset = SST2Dataset(train_data, tokenizer)
        test_dataset = SST2Dataset(test_data, tokenizer)

    if task == 'qqp':
        from dataset.qqpdataset import QQPDataset

        dataset = load_dataset('glue', 'qqp')
        num_classes = 2
        batch_size = 400

        train_data = dataset['train']
        test_data = dataset['validation']
        train_dataset = QQPDataset(train_data, tokenizer)
        test_dataset = QQPDataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(num_classes=num_classes, task=task, training_type=training_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    wandb.watch(model, log='all')

    for epoch in range(epochs):
        train_loss, train_accuracy, train_time_per_batch = model.train_epoch(train_dataloader, optimizer, device)
        test_loss, test_accuracy, test_time_per_batch = model.test_epoch(test_dataloader, device)

        wandb.log(
            {
                'Epoch':epoch,
                'Train Loss':train_loss,
                'Train Accuracy':train_accuracy,
                'Train Time':train_time_per_batch,
                'Test Loss':test_loss,
                'Test Accuracy':test_accuracy,
                'Test Time':test_time_per_batch
            }
        )
        
        print(f'Epoch {epoch + 1} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}')
    print(f'{model.trained_proportion * 100}% of the model was trained')

    if training_type == 'finetuned': 
        file_location = model.file_write()
        wandb.save(file_location)
    wandb.finish()