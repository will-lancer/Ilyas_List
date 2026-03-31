import torch
import torch.nn.functional as F
import random

words = open('../names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

block_size = 4

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
emb_dim = 8
nn_size = 80
sample_len = emb_dim * block_size

# params

C = torch.randn((VOCAB_LEN, emb_dim), generator=g)
W1 = torch.randn((sample_len, nn_size), generator=g) * (2 / sample_len) ** 0.5
b1 = torch.zeros(nn_size)
W2 = torch.randn((nn_size, VOCAB_LEN), generator=g)
b2 = torch.randn(VOCAB_LEN, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True


batch_size = 32
decay_rate = 10**(-5)

for _ in range(10**6):

    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,)) # min, max, size

    # forward pass
    emb = C[Xtr[ix]]
    h = F.relu(emb.view(-1, sample_len) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[ix])

    if (_ % 1000 == 0): print(f'Training loss: {loss}')

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # dev loss
    if (_ % 1000 == 0):
        with torch.no_grad():
            emb_dev = C[Xdev]
            h_dev = F.relu(emb_dev.view(-1, sample_len) @ W1 + b1)
            logits_dev = h_dev @ W2 + b2
            dev_loss = F.cross_entropy(logits_dev, Ydev)
            print(f'Dev loss: {dev_loss}')

    # update
    decay_func = 1/(1 + _ * decay_rate)
    lr = 0.1 * decay_func
    # lr = 0.1 if _ < 1000 else 0.05
    for p in parameters:
        p.data += - lr * p.grad
    

# Hyperparameter and run log

# Test run 1:
# - emb_dim: 2
# - nn_size: 100
# - batch_size: 32
# - training iterations: 1000
# - learning rate: 0.1
# - Final loss: 2.68


# Test run 2:
# - emb_dim: 2
# - nn_size: 100
# - batch_size: 10,000
# - training iterations: 1000
# - learning rate: 0.1
# - Final loss: 2.567

# bigger batch size is slightly better

# Test run 3:
# - emb_dim: 2
# - nn_size: 100
# - batch_size: 32
# - training iterations: 3000
# - learning rate: 0.1
# - Final loss: 2.567

# running it for longer is better too

# Test run 4:
# - emb_dim: 2
# - nn_size: 100
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1
# - Final loss: 2.47

# longer == better basically

# Test run 5:
# - emb_dim: 10
# - nn_size: 100
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.5

# 10 emb_dim is worse than 2

# Test run 6:
# - emb_dim: 5
# - nn_size: 100
# - batch_size: 32
# - training iterations: 3000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.54

# but five may be better than two

# Test run 7:
# - emb_dim: 5
# - nn_size: 100
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.43

# yep! 

# Test run 7:
# - emb_dim: 5
# - nn_size: 60
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.4

# smaller layer size may be better

# Test run 8:
# - emb_dim: 5
# - nn_size: 40
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.38

# Test run 9:
# - emb_dim: 5
# - nn_size: 30
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.39

# not too small

# Test run 10:
# - emb_dim: 5
# - nn_size: 50
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.39

# seems like 40 was the best

# Test run 11:
# - block_size = 4 (was 3 for all previous trials)
# - emb_dim: 5
# - nn_size: 50
# - batch_size: 32
# - training iterations: 10000
# - learning rate: 0.1 --> 0.05
# - Final loss: 2.37

# Test run 12:
# - block_size = 4 (was 3 for all previous trials)
# - emb_dim: 5
# - nn_size: 50
# - batch_size: 32
# - training iterations: 10000
# - learning rate: Used decay function w decay_rate = 10**(-8)
# - Final loss: 2.366

# Test run 13:
# - block_size = 4 (was 3 for all previous trials)
# - emb_dim: 5
# - nn_size: 50
# - batch_size: 32
# - training iterations: around 100k
# - learning rate: Used decay function w decay_rate = 10**(-5)
# - Final loss: 2.28

# before ReLU best: 2.14, after: 2.13. Beat Karpathy by 0.03. Boom. Done.