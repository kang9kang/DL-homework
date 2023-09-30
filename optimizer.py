class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, model):
        for p in model.parameters():
            p.data -= p.grad.data * self.lr