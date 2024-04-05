import torch
import copy
import logging
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# Configure Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Learner:
    def __init__(self, config, model, tokenizer, train_dataloader, val_dataloader, loss_func, optimizer, scheduler = None, writer = None, device = 'cpu'):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = 0
        self.device = device
        self.cur_step = 1
        self.best_val_loss = float('inf')
        self.writer = writer
        self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
    
    def validation_epoch(self):
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(self.val_dataloader):
                if idx == 1:
                    break
                encoder_input = batch[0].to(self.device)
                decoder_input = batch[1].to(self.device)
                
                for i in range(encoder_input.size(0)):
                    if i == 8:
                        break
                    source_text = self.tokenizer.decode(encoder_input[i].tolist(), skip_special_tokens = False)
                    target_text = self.tokenizer.decode(decoder_input[i].tolist(), skip_special_tokens = False)
                    predicted_text = self.model.translate(source_text, self.tokenizer)
                    print('-' * 80)
                    print(f"Source: {source_text}")
                    print(f"Target: {target_text}")
                    print(f"Predicted: {predicted_text}")

    def training_epoch(self, epoch):
        self.model.train()
        batch_iterator = tqdm(self.train_dataloader, desc = f'Processing Epoch {epoch:02d}', total = len(self.train_dataloader))
        loss_sum = 0
        for batch in batch_iterator:
            encoder_input = batch[0].to(self.device)
            decoder_input = batch[1].to(self.device)
            pred_token_ids = self.model(encoder_input, decoder_input)
            loss = self.loss_func(
                pred_token_ids.reshape(-1, pred_token_ids.size(-1)), # Reshaping for loss
                decoder_input[:, 1:].contiguous().view(-1) # Shifting right (without BOS)
            )
            loss_sum += loss.item()
            batch_iterator.set_postfix({'loss': f"{loss.item():6.3f}"})
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', loss.item(), self.global_step)
                self.writer.flush()
                
            loss.backward()
            if self.cur_step % self.config.GRAD_ACCUMULATION_STEPS == 0:
                self.optimizer.step()
                if self.scheduler != None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            self.cur_step += 1
            self.global_step += 1
            
        loss_avg = loss_sum / len(self.train_dataloader)
        # Save model every 'epoch_cnt' epochs
        if epoch % self.config.MODEL_SAVE_EPOCH_CNT == 0:
            epoch_ckpt_pth = os.path.join(self.config.SAVE_MODEL_DIR, f'model_ckpt_epoch{epoch}.pt')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, epoch_ckpt_pth)
            print('    - [Info] The checkpoint file has been updated.')
    
        # Save best model
        if loss_avg < self.best_val_loss:
            self.best_val_loss = loss_avg
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict()) 
        

    def fit(self, start_epoch, n_epochs):
        self.n_epochs = n_epochs
        for epoch_idx in range(start_epoch, start_epoch + n_epochs):
            self.training_epoch(epoch_idx)
            self.validation_epoch()
        # Save best model
        best_model_ckpt_pth = os.path.join(self.config.SAVE_MODEL_DIR, f'model_ckpt_best.pt')
        best_checkpoint = {
            'model_state_dict': self.best_model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(best_checkpoint, best_model_ckpt_pth)  
        print('    - [Info] The best checkpoint file has been updated.') 
            