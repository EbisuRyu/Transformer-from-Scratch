import os
import torch
import numpy as np

class CheckpointSaver:
    def __init__(self, dirpath, decreasing = True, top_n = 5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model, optimizer, epoch, metric_val):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt')
        save = metric_val < self.best_metric_val if self.decreasing else metric_val > self.best_metric_val
        if save: 
            print(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}")
            self.best_metric_val = metric_val
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, model_path)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key = lambda o: o['score'], reverse = not self.decreasing)
        if len(self.top_model_paths) > self.top_n: 
            self.cleanup()
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n : ]
        print(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[ : self.top_n]
