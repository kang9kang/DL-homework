from tensor import MyTensor
import numpy as np
import function as F1
import torch.nn.functional as F2
import torch.nn as nn
import torch
from layer import MyLayer, MyLinear, MySigmoid, MyReLu, MyTanh, MyCrossEntropyLoss, MySequential, MyBatchNorm
from dataloader import HalfMoonDataSet, MyDataLoader
from torch.utils.data import DataLoader

class Net(MyLayer):
    def __init__(self):
        super().__init__()
        self.net = MySequential(MyLinear(2, 2), MySigmoid(), MyLinear(2, 2), MySigmoid(), MyLinear(2, 2), MySigmoid())
        
    def forward(self, X: MyTensor) -> MyTensor:
        return self.net(X)
    
    def backward(self, eta: np.ndarray) -> np.ndarray:
        return self.net.backward(eta)
    
    def update(self, lr: float) -> None:
        self.net.update(lr)

if __name__ == "__main__":
    np.random.seed(59)
    
    train_dataset = HalfMoonDataSet(n_samples=7000, noise=0.05, random_state=0)
    train_dataloader = MyDataLoader(train_dataset, batch_size=16, shuffle=True)
    
    fc = MyBatchNorm(2 ,True)
    fc.gamma = MyTensor(np.random.randn(2)) * 0.01
    fc.beta = MyTensor(np.random.randn(2)) * 0.01
    
    weight2 = torch.tensor(fc.gamma.view(np.ndarray), requires_grad=True, dtype=torch.double)
    bias2 = torch.tensor(fc.beta.view(np.ndarray), requires_grad=True, dtype=torch.double)
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        x1 = data
        x2 = torch.tensor(data.view(np.ndarray), requires_grad=True, dtype=torch.double)
        y1 = fc(x1)
        print(y1)
        y2 = F2.batch_norm(x2, torch.zeros(2, dtype=torch.double), torch.ones(2, dtype=torch.double), weight2, bias2, True, 0.1, 1e-5)
        print(y2)
        z1 = fc.backward(np.ones_like(y1))
        print(fc.gamma.grad)
        y2.sum().backward()
        print(weight2.grad)
        break
