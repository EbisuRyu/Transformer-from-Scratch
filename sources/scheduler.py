class CustomScheduler():
    def __init__(self, optimizer, d_model, n_warmup_steps=4000):
        # Initialize the scheduler
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.cur_step = 0
        self.cur_lr = None

        # Initialize LR right away
        self.step()

    def step(self):
        # Perform one optimization step
        self.cur_step += 1
        self.cur_lr = self._get_lr()

        # Update learning rate in optimizer
        for p in self.optimizer.param_groups:
            p['lr'] = self.cur_lr

    def _get_lr(self):
        # Calculate learning rate based on current step
        return self.d_model**(-0.5) * min(self.cur_step**(-0.5), self.cur_step * self.n_warmup_steps**(-1.5))

    def get_last_lr(self):
        # Get the last learning rate used for optimization
        return [group['lr'] for group in self.optimizer.param_groups]

    def zero_grad(self):
        # Zeroes the gradient buffers of all parameters
        self.optimizer.zero_grad()
