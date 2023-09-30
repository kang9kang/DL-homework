from tensor import MyTensor
import numpy as np
import function as F1
import torch.nn.functional as F2
import torch.nn as nn
import torch
from layer import MyLayer, MyLinear, MySigmoid, MyReLu, MyTanh, MyCrossEntropyLoss
from dataloader import HalfMoonDataSet, DataLoader

class Net(MyLayer):
    def __init__(self):
        super().__init__()
        self.net = [MyLinear(2, 25), MyTanh(), MyLinear(25, 25), MyTanh(), MyLinear(25, 2)]

if __name__ == "__main__":
    # np.random.seed(59)
    # a1 = MyTensor(np.random.randn(4, 4))
    # b1 = MyTensor(np.random.randn(4, 2))
    # c1 = MyTensor(np.random.randn(2))
    # d1 = MyTensor([[0,1],[1,0],[0,1],[1,0]])
    # a2 = torch.tensor(a1.view(np.ndarray), dtype=torch.float32, requires_grad=True)
    # b2 = torch.tensor(b1.view(np.ndarray), dtype=torch.float32, requires_grad=True)
    # c2 = torch.tensor(c1.view(np.ndarray), dtype=torch.float32, requires_grad=True)
    # d2 = torch.tensor(d1.view(np.ndarray), dtype=torch.float32)
    # fc1 = MyLinear(4, 2)
    # fc1.W = b1
    # fc1.b = c1
    # y1 = fc1(a1)
    # print(y1.shape)
    # fc2 = MyCrossEntropyLoss()
    # z5 = fc2(y1, d1)
    # # print(z1)
    # z6 = fc2.backward(1)
    # z7 = fc1.backward(z6)

    # z2 = F2.linear(a2, b2.T, c2)
    # # print(z2)
    # z3 = F2.cross_entropy(z2, d2)
    # z3.backward()

    # print(z7)
    # print(a2.grad)

    train_dataset = HalfMoonDataSet(n_samples=7000, noise=0.05, random_state=0)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    fc = MyLinear(2, 2)
    # fc2 = MyLinear(2, 2)
    losslayer = MyCrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(target)
        y = fc(data)
        print(y)
        loss = losslayer(y, target)
        print(loss)
        z = losslayer.backward(1)
        fc.backward(z)
        print(fc.W)
        print(fc.b)
        fc.update(0.01)
        print(fc.W)
        print(fc.b)
        break
