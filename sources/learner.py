import torch
import copy
import logging
import os
import wandb
import torch.nn as nn

from tqdm import tqdm
from .utils import CheckpointSaver
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Configure logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Learner:
    def __init__(self, config, model, tokenizer, train_dataloader, val_dataloader, loss_func, optimizer, scheduler=None, device='cpu'):
        # Initialize Learner with required components
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Initialize CheckpointSaver for saving model checkpoints
        self.checkpointSaver = CheckpointSaver(os.path.join(self.config.RUNS_FOLDER_PTH, self.config.RUN_NAME), decreasing=True, top_n=5)
        self.training_step = 1
        self.global_step = 1

    def track_example(self, epoch, num_examples=2):
        # Track example translations during training and validation
        self.model.eval()
        with torch.no_grad():
            # Training examples
            train_batch = next(iter(self.train_dataloader))
            train_encoder_input = train_batch[0].to(self.device)
            train_decoder_input = train_batch[1].to(self.device)
            for i in range(train_encoder_input.size(0)):
                if i == num_examples:
                    break
                train_source_text = self.tokenizer.decode(train_encoder_input[i].tolist(), skip_special_tokens=False)
                train_target_text = self.tokenizer.decode(train_decoder_input[i].tolist(), skip_special_tokens=False)
                train_predicted_text = self.model.translate(train_source_text, self.tokenizer)
                # Print and log training examples
                print('-' * 80)
                print(f"Epoch: {epoch}")
                print(f'Data: Train')
                print(f"Source: {train_source_text}")
                print(f"Target: {train_target_text}")
                print(f"Predicted: {train_predicted_text}")
                self.table.add_data(epoch, 'Train', train_source_text, train_target_text, train_predicted_text)

            # Validation examples
            val_batch = next(iter(self.val_dataloader))
            val_encoder_input = val_batch[0].to(self.device)
            val_decoder_input = val_batch[1].to(self.device)
            for i in range(val_encoder_input.size(0)):
                if i == num_examples:
                    break
                val_source_text = self.tokenizer.decode(val_encoder_input[i].tolist(), skip_special_tokens=False)
                val_target_text = self.tokenizer.decode(val_decoder_input[i].tolist(), skip_special_tokens=False)
                val_predicted_text = self.model.translate(val_source_text, self.tokenizer)
                # Print and log validation examples
                print('-' * 80)
                print(f"Epoch: {epoch}")
                print(f'Data: Validation')
                print(f"Source: {val_source_text}")
                print(f"Target: {val_target_text}")
                print(f"Predicted: {val_predicted_text}")
                self.table.add_data(epoch, 'Validation', val_source_text, val_target_text, val_predicted_text)

    def validation_epoch(self, epoch):
        # Validation phase for calculating loss and BLEU score
        self.model.eval()
        predict_text_list = []
        target_text_list = []
        loss_sum = 0
        with torch.no_grad():
            batch_iterator = tqdm(self.val_dataloader, desc=f'Validation Epoch {epoch:02d}', total=len(self.val_dataloader))
            for batch_idx, batch in enumerate(batch_iterator):
                encoder_input = batch[0].to(self.device)
                decoder_input = batch[1].to(self.device)
                pred_token_ids = self.model(encoder_input, decoder_input)
                # Compute loss
                loss = self.loss_func(
                    pred_token_ids.reshape(-1, pred_token_ids.size(-1)),  # Reshaping for loss
                    decoder_input[:, 1:].contiguous().view(-1)  # Shifting right (without BOS)
                )
                loss_sum += loss.item()
                # Logging loss
                wandb.log({
                    'Epoch': epoch,
                    'Validation/Loss': loss.item()
                }, step=self.global_step)
                self.global_step += 1

                # Process predicted tokens
                pred_token_ids = pred_token_ids.detach().cpu()
                pred_token_ids = nn.functional.log_softmax(pred_token_ids, dim=-1)
                pred_token_ids = pred_token_ids.argmax(dim=-1).squeeze(-1)

                # Decode predicted and target texts
                predict_text = self.tokenizer.decode_batch(pred_token_ids.numpy(), skip_special_tokens=False)
                target_text = self.tokenizer.decode_batch(decoder_input.detach().cpu().numpy(), skip_special_tokens=False)

                # Split texts into tokens
                predict_text = [sentence.split() for sentence in predict_text]
                target_text = [[sentence.split()] for sentence in target_text]

                predict_text_list += predict_text
                target_text_list += target_text

            # Compute BLEU score
            loss_avg = loss_sum / len(self.val_dataloader)
            bleu_score = corpus_bleu(target_text_list, predict_text_list, smoothing_function=SmoothingFunction().method4)
            # Logging BLEU score and loss
            wandb.log(
                {
                    'Epoch': epoch,
                    'Validation/BLEU': bleu_score,
                    'Validation/Avg_loss': loss_avg
                }, step=self.global_step)
            self.global_step += 1

            print(f'    - [Info] Validation Loss: {loss_avg:.3f}, BLEU Score: {bleu_score:.3f}')

    def training_epoch(self, epoch):
        # Training phase
        self.model.train()
        batch_iterator = tqdm(self.train_dataloader, desc=f'Processing Epoch {epoch:02d}', total=len(self.train_dataloader))
        loss_sum = 0
        for batch_idx, batch in enumerate(batch_iterator):
            encoder_input = batch[0].to(self.device)
            decoder_input = batch[1].to(self.device)
            pred_token_ids = self.model(encoder_input, decoder_input)
            # Compute loss
            loss = self.loss_func(
                pred_token_ids.reshape(-1, pred_token_ids.size(-1)),  # Reshaping for loss
                decoder_input[:, 1:].contiguous().view(-1)  # Shifting right (without BOS)
            )
            loss_sum += loss.item()
            # Update progress bar with loss
            batch_iterator.set_postfix({'loss': f"{loss.item():6.3f}"})
            loss.backward()
            if self.training_step % self.config.GRAD_ACCUMULATION_STEPS == 0:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            # Logging training loss
            wandb.log({
                'Epoch': epoch,
                'Batch': batch_idx,
                'Train/Loss': loss.item()
            }, step=self.global_step)
            self.global_step += 1
            self.training_step += 1

        # Compute and log average training loss
        loss_avg = loss_sum / len(self.train_dataloader)
        wandb.log({
            'Epoch': epoch,
            'Train/Avg_loss': loss_avg
        }, step=self.global_step)
        self.global_step += 1

        # Save model checkpoint every specified number of epochs
        if epoch % self.config.MODEL_SAVE_EPOCH_CNT == 0:
            self.checkpointSaver(self.model, self.optimizer, epoch, loss_avg)
            print('    - [Info] The checkpoint file has been updated.')

    def fit(self, start_epoch, n_epochs):
        # Train the model for specified epochs
        self.table = wandb.Table(columns=["Epoch", "Source", "Data", "Target", "Predicted"])
        self.n_epochs = n_epochs

        for epoch_idx in range(start_epoch, start_epoch + n_epochs):
            self.track_example(epoch_idx, num_examples=2)
            self.validation_epoch(epoch_idx)
            self.training_epoch(epoch_idx)

        wandb.log({'Tracking': self.table})  # Log the table