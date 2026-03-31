import torch
import torch.nn.functional as F
import random

words = open('../names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

block_size = 3

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]

            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# splitting up data into train, dev, and test sets
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])

# manual seed to match andrej
g = torch.manual_seed(2147483647)

# param vars 

VOCAB_LEN = 27
n_emb = 12
n_hidden = 200
sample_len = n_emb * block_size

# nn params

C = torch.randn((VOCAB_LEN, n_emb), generator=g)
W1 = torch.randn((sample_len, n_hidden), generator=g) * (5/3) / (n_emb * block_size)**(0.5)
W2 = torch.randn((n_hidden, VOCAB_LEN), generator=g) * 0.01
b2 = torch.randn(VOCAB_LEN, generator=g) * 0

# batchNorm params

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running = torch.ones((1, n_hidden))

parameters = [C, W1, W2, b2, bngain, bnbias]
for p in parameters:
    p.requires_grad = True


max_steps = 10**6
batch_size = 32

for _ in range(max_steps):

    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator = g) # min, max, size
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward pass
    emb = C[Xb]
    embcat = emb.view(emb.shape[0], -1)

    # linear layer
    hpreact = embcat @ W1

    # batchNorm layer
    bnmeani = hpreact.mean(0, keepdim = True)
    bnstdi = hpreact.std(0, keepdim = True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
    
    # non-linearity
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yb)

    if (_ % 1000 == 0): print(f'Training loss: {loss}')

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # dev loss
    if (_ % 1000 == 0):
        with torch.no_grad():
            emb_dev = C[Xdev]
            hpre_dev = emb_dev.view(-1, sample_len) @ W1
            hpre_dev = bngain * (hpre_dev - bnmean_running) / bnstd_running + bnbias
            h_dev = torch.tanh(hpre_dev)
            logits_dev = h_dev @ W2 + b2
            dev_loss = F.cross_entropy(logits_dev, Ydev)
            print(f'Dev loss: {dev_loss}')

    # update
    lr = 0.1 if _ < 100000 else 0.01
    for p in parameters:
        p.data += - lr * p.grad
    

# Hyperparameter and run log

# Test run 1:
# - n_emb: 2
# - n_hidden: 100
# - batch_size: 32
# - training iterations: 1000
# - learning rate: 0.1
# - Final loss: 2.68


# Test run 2:
# - n_emb: 2
# - n_hidden: 100
# - batch_size: 10,000
# - training iterations: 1000
# - learning rate: 0.1
# - Final loss: 2.567

# bigger batch size is slightly better

# Test run 3:
# - n_emb: 2
# - n_hidden: 100
# - batch_size: 32
# - training iterations: 3000
# - learning rate: 0.1
# - Final loss: 2.567

# running it for longer is better too

# Test run 4:
# - n_emb: 2
# - n_hidden: 100
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1
# - Final loss: 2.47

# longer == better basically

# Test run 5:
# - n_emb: 10
# - n_hidden: 100
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.5

# 10 n_emb is worse than 2

# Test run 6:
# - n_emb: 5
# - n_hidden: 100
# - batch_size: 32
# - training iterations: 3000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.54

# but five may be better than two

# Test run 7:
# - n_emb: 5
# - n_hidden: 100
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.43

# yep! 

# Test run 7:
# - n_emb: 5
# - n_hidden: 60
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.4

# smaller layer size may be better

# Test run 8:
# - n_emb: 5
# - n_hidden: 40
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.38

# Test run 9:
# - n_emb: 5
# - n_hidden: 30
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.39

# not too small

# Test run 10:
# - n_emb: 5
# - n_hidden: 50
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.39

# seems like 40 was the best

# Test run 11:
# - block_size = 4 (was 3 for all previous trials)
# - n_emb: 5
# - n_hidden: 50
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.37

# Test run 12:
# - block_size = 4 (was 3 for all previous trials)
# - n_emb: 5
# - n_hidden: 50
# - batch_size: 32
# - training iterations: 10000
# - learning rate: Used decay function w decay_rate = 10**(-8)
# - Final loss: 2.366

# Test run 13:
# - block_size = 4 (was 3 for all previous trials)
# - n_emb: 5
# - n_hidden: 50
# - batch_size: 32
# - training iterations: around 100k
# - learning rate: Used decay function w decay_rate = 10**(-5)
# - Final loss: 2.28

# before ReLU best: 2.14, after: 2.13. Beat Karpathy by 0.03. Boom. Done.