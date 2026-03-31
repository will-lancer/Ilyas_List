# E02: randomly split the dataset into 80% train, 10% dev, and 10% test. Train 
# both bigram and trigram models only on the train set. Evaluate them on the dev 
# and test splits. What do you observe?
# E03: use the dev set to tune the smoothing (regularization) strength for the 
# trigram model—try several values and see which works best based on dev loss. 
# What patterns do you notice in train/dev loss as you tune? Take the best 
# smoothing setting and evaluate once on the test set at the end. What loss do 
# you achieve?
# E04: since our 1-hot vectors just select a row of W, making them explicitly 
# is wasteful. Can you eliminate F.one_hot in favor of simply indexing rows of W?
# E05: use F.cross_entropy instead; the result should be identical. Why might we 
# prefer F.cross_entropy?


# E01: train a trigram language model, i.e. take two characters as input to 
# predict the 3rd one. Use either counting or a neural net. Evaluate the loss. 
# Did it improve over a bigram model?

import torch
import torch.nn.functional as F

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

num = 0
xs, ys = [], []

count = 0

for w in words:
    w = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(zip(w, w[1:]), w[2:]):
        ix1 = stoi[ch1[0]]
        ix2 = stoi[ch1[1]]
        xInd = ix1 * 27 + ix2

        ix3 = stoi[ch2]
        yInd = ix3

        xs.append(xInd)
        ys.append(yInd)
        count += 1
        
xs = torch.tensor(xs)
ys = torch.tensor(ys)

W = torch.randn((27*27, 27), requires_grad = True)
xenc = F.one_hot(xs, num_classes = 27 * 27).float()

def softmax(logits):
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims = True)
    return probs

for i in range(1000):

    # forward pass
    logits = xenc @ W
    probs = softmax(logits)
    loss = - probs[torch.arange(count), ys].log().mean()

    # backward pass
    W.grad = None
    loss.backward()

    if (i < 300):
        W.data += - 100 * W.grad
    else:
        W.data += - 50 * W.grad

    if (i % 100 == 0):
        print(loss.item())

for i in range(5):

    out = []
    ix1, ix2 = 0, 0

    while True:
        pair_index = ix1 * 27 + ix2
        xenc = F.one_hot(torch.tensor([pair_index]), num_classes=27*27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        ix3 = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(itos[ix3])
        ix1, ix2 = ix2, ix3
        if ix3 == 0:
            break

    print(''.join(out))

# SUCCESS!!!!!!


# ----------------Land of Failed Ideas------------------- #


# # first very stupid idea: just do the exact same thing
# # for i in range(10):

# #     # forward pass
# #     xenc = F.one_hot(xs, num_classes = 27).float()
# #     logits = xenc @ W
# #     loss = loss(logits)

# #     # backward pass
# #     W.grad = None
# #     loss.backward()
# #     W.data += - 1 * W.grad

# #     print(loss)

# # this did not work because of the obvious problem of matrix multiplication. i asked
# # chat and it suggested flattening the two onehots into a matrix using view

# # second stupid idea: put the one hots on top of each other

# W = torch.randn((27*27, 27), requires_grad = True)

# count = 0

# for i in range(100):
#     # forward pass
#     xenc = F.one_hot(xs, num_classes = 27).float()
#     logits = xenc @ W
#     probs = softmax(logits)
#     loss = - probs[torch.arange(num), ys].log().mean()

#     # backward pass
#     W.grad = None
#     loss.backward()
#     W.data += - 1 * W.grad
#     count += 1