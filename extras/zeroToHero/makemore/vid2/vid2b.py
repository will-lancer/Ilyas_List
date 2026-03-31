import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

# N = torch.zeros((27, 27), dtype = torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

xs, ys = [], []

num = 0
for w in words:
    w = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(w, w[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        xs.append(ix1)
        ys.append(ix2)
        num += 1

xs = torch.tensor(xs)
ys = torch.tensor(ys)

g = torch.manual_seed(2147483647)
W = torch.randn((27, 27), generator = g, requires_grad = True)


def softmax(logits):
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims = True)
    return probs


count = 0

for i in range(2):
    # forward pass
    xenc = F.one_hot(xs, num_classes = 27).float()
    logits = xenc @ W
    print(logits.shape)
    probs = softmax(logits)
    loss = - probs[torch.arange(num), ys].log().mean()

    # backward pass
    W.grad = None
    loss.backward()

    W.data += - 10 * W.grad

    if (count % 5 == 0):
        print(loss.item())