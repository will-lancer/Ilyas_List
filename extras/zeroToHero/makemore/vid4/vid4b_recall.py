import torch
import torch.nn.functional as F
import random
from pathlib import Path

from makemore.vid4.vid4b import vocab_size

# goal: build an MLP to predict the next character in a sequence given the past n-characters
# in a sequence

# importing data — anchored to this file so debugger cwd (often workspace root) still works
_DATA = Path(__file__).resolve().parent.parent / "names.txt"
words = _DATA.read_text().splitlines()

chars = sorted(set(list(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# building out dataset of character n-grams

block_size = 3

def build_dataset(words):
    X, Y = [], []
    
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            i = stoi[ch]

            X.append(context)
            Y.append(i)

            context = context[1:] + [i]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y

# make train, test, dev sets

X, Y = build_dataset(words)

random.seed(42)
random.shuffle(words)

n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = X[:n1], Y[:n1]
Xdev, Ydev = X[n1:n2], Y[n1:n2]
Xtest, Ytest = X[n2:], Y[n2:]

# build out layers

g = torch.manual_seed(2147483647)

class Linear:

    def __init__(self, fan_in, fan_out, has_bias = True):
        self.weight = torch.randn((fan_in, fan_out), generator = g) / fan_out**0.5
        self.bias = torch.zeros((fan_out)) if has_bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class BatchNorm1D:

    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        # basics
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # params
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # buffers
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdims = True)
            xvar = x.var(0, keepdims = True, unbiased = False)
        else:
            xmean = self.running_mean
            xvar = self.runnning_var
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # buffers
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

    
class Tanh:

    def __init__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        # there are no parameters here, but want to include parameters
        # in every 'pytorch' class
        return []

# training nn

n_emb = 10
n_hidden = 200
vocab_size = len(stoi)

# todo later








        
