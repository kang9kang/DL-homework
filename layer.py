import numpy as np
from tensor import MyTensor
import function as F


class MyLayer:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self) -> MyTensor:
        pass

    def backward(self) -> np.ndarray:
        pass

    def update(self, lr: float) -> None:
        pass


class MyLinear(MyLayer):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(MyLinear, self).__init__()
        self.W = MyTensor(np.random.randn(in_features, out_features)) * 0.01
        self.b = MyTensor(np.random.randn(out_features)) * 0.01

    def forward(self, X: MyTensor) -> MyTensor:
        self.X = X.view(np.ndarray).reshape(-1, X.shape[-1])
        y = F.linear(X, self.W, self.b)
        return y

    def backward(self, eta: np.ndarray) -> np.ndarray:
        self.W.grad = np.dot(self.X.T, eta.reshape(-1, eta.shape[-1]))
        self.b.grad = eta.reshape(-1, eta.shape[-1]).sum(axis=0)
        return np.dot(eta, self.W.T)

    def update(self, lr: float) -> None:
        self.W -= self.W.grad * lr
        self.b -= self.b.grad * lr

class MySigmoid(MyLayer):
    def __init__(self):
        super(MySigmoid, self).__init__()

    def forward(self, X: MyTensor) -> MyTensor:
        y = F.sigmoid(X)
        self.y = y.view(np.ndarray)
        return y

    def backward(self, eta: np.ndarray) -> np.ndarray:
        return eta * self.y * (1 - self.y)


class MyReLu(MyLayer):
    def __init__(self):
        super(MyReLu, self).__init__()

    def forward(self, X: MyTensor) -> MyTensor:
        self.X = X.view(np.ndarray)
        return F.relu(X)

    def backward(self, eta: np.ndarray) -> np.ndarray:
        return eta * (self.X > 0)


class MyTanh(MyLayer):
    def __init__(self):
        super(MyTanh, self).__init__()

    def forward(self, X: MyTensor) -> MyTensor:
        y = F.tanh(X)
        self.y = y.view(np.ndarray)
        return y

    def backward(self, eta: np.ndarray) -> np.ndarray:
        return eta * (1 - self.y**2)


    
class MyCrossEntropyLoss(MyLayer):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()

    def forward(self, y: MyTensor, t: MyTensor) -> MyTensor:
        y = F.softmax(y)
        self.y = y.view(np.ndarray)
        y = F.cross_entropy(y, t)
        self.t = t.view(np.ndarray)
        return y

    def backward(self, eta=1) -> np.ndarray:
        return eta * (self.y - self.t) / len(self.y)

