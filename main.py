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
        learning_rate, 
        num_classes, 
        batch_size,
        dropout,
        max_tokens,
        weight_decay, 
        betas,
        eps,
):

    tokenizer = AutoTokenizer.from_pretrained(model)
    task = task
    training_type = training_type
    log_to_wandb = log_to_wandb
    epochs = epochs
    learning_rate = learning_rate
    batch_size = batch_size
    num_classes = num_classes
    max_tokens = max_tokens

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
        num_classes = num_classes
        batch_size = batch_size
        max_tokens = max_tokens

        train_data = dataset['train']
        test_data = dataset['validation']
        train_dataset = SST2Dataset(train_data, tokenizer)
        test_dataset = SST2Dataset(test_data, tokenizer)

    if task == 'qqp':
        from dataset.qqpdataset import QQPDataset

        dataset = load_dataset('glue', 'qqp')
        num_classes = num_classes
        batch_size = batch_size

        train_data = dataset['train']
        test_data = dataset['validation']
        train_dataset = QQPDataset(train_data, tokenizer)
        test_dataset = QQPDataset(test_data, tokenizer)

    if task == 'cola':
        from dataset.coladataset import COLADataset

        dataset = load_dataset('glue', 'cola')
        num_classes = num_classes
        batch_size = batch_size

        train_data = dataset['train']
        test_data = dataset['validation']
        train_dataset = COLADataset(train_data, tokenizer)
        test_dataset = COLADataset(test_data, tokenizer)

    if task == 'wnli':
        from dataset.wnlidataset import WNLIDataset

        dataset = load_dataset('glue', 'wnli')
        num_classes = num_classes
        batch_size = batch_size

        train_data = dataset['train']
        test_data = dataset['validation']
        train_dataset = WNLIDataset(train_data, tokenizer)
        test_dataset = WNLIDataset(test_data, tokenizer)

    if task == 'stsb':
        from dataset.stsbdataset import STSBDataset

        dataset = load_dataset('glue', 'stsb')
        num_classes = num_classes
        batch_size = batch_size

        train_data = dataset['train']
        test_data = dataset['validation']
        train_dataset = STSBDataset(train_data, tokenizer)
        test_dataset = STSBDataset(test_data, tokenizer)

    if task == 'rte':
        from dataset.rtedataset import RTEDataset

        dataset = load_dataset('glue', 'rte')
        num_classes = num_classes
        batch_size = batch_size

        train_data = dataset['train']
        test_data = dataset['validation']
        train_dataset = RTEDataset(train_data, tokenizer)
        test_dataset = RTEDataset(test_data, tokenizer)
    
    if task == 'mrpc':
        from dataset.mrpcdataset import MRPCDataset

        dataset = load_dataset('glue', 'mrpc')
        num_classes = num_classes
        batch_size = batch_size

        train_data = dataset['train']
        test_data = dataset['validation']
        train_dataset = MRPCDataset(train_data, tokenizer)
        test_dataset = MRPCDataset(test_data, tokenizer)

    if task == 'qnli':
        from dataset.qnlidataset import QNLIDataset

        dataset = load_dataset('glue', 'qnli')
        num_classes = num_classes
        batch_size = batch_size

        train_data = dataset['train']
        test_data = dataset['validation']
        train_dataset = QNLIDataset(train_data, tokenizer)
        test_dataset = QNLIDataset(test_data, tokenizer)

    if task == 'mnli':
        from dataset.mnlidataset import MNLIDataset

        dataset = load_dataset('glue', 'mnli')
        num_classes = num_classes
        batch_size = batch_size

        train_data = dataset['train']
        test_data = dataset['validation_matched']
        train_dataset = MNLIDataset(train_data, tokenizer)
        test_dataset = MNLIDataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    model = Model(num_classes=num_classes, task=task, training_type=training_type, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
    wandb.watch(model, log='all')

    for epoch in range(epochs):
        train_loss, train_accuracy, train_loss_list, train_accuracy_list, train_time_list, train_step_list = model.train_epoch(train_dataloader, optimizer, device)
        test_loss, test_accuracy, test_loss_list, test_accuracy_list, test_time_list, test_step_list = model.test_epoch(test_dataloader, device)

        for tr_loss, tr_accuracy, tr_time, tr_step in zip(train_loss_list, train_accuracy_list, train_time_list, train_step_list):
            wandb.log(
                {
                    'Train Loss':tr_loss,
                    'Train Accuracy':tr_accuracy,
                    'Train Time':tr_time,
                }, 
                step = (epoch * len(train_step_list)) + tr_step
            )
        for te_loss, te_accuracy, te_time, te_step in zip(test_loss_list, test_accuracy_list, test_time_list, test_step_list):
            wandb.log(
                {
                    'Test Loss':te_loss,
                    'Test Accuracy':te_accuracy,
                    'Test Time':te_time,
                }, 
                step = (epoch * len(test_step_list)) + te_step 
            )
            import pdb; pdb.set_trace()
 
        print(f'Epoch {epoch + 1} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}')
    print(f'{model.trained_proportion * 100}% of the model was trained')

    if training_type == 'finetuned': 
        file_location = model.file_write()
        wandb.save(file_location)
    wandb.finish()