import torch
import numpy as np
import matplotlib.pyplot as plt


words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27, 27), dtype = torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

P = N.float()
P /= P.sum(1, keepdim = True) # broadcasting ftw!

# warning from andrey: make sure you get your broadcasting right...

for w in words:
    w = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(w, w[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        
        N[ix1, ix2] += 1

g = torch.Generator().manual_seed(2147483647)

# 27, 27
# 27,  1

# this is kind of like the Young Tableaux rules for taking tensor products

for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples = 1, replacement = True, generator = g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

"""
ok, so actual structure of makemore is this: you make a frequency table of the pairs of
chars from the wds. you make maps to and from the character index to the index itself, e.g.
a <--> 1, . <--> 0, z <--> 27, etc. then you run a while loop that does the following:
it makes an empty list of characters and intializes an index ix = 0. then it pulls a row
from your array, normalizes it into probabilities, samples another random index from this
array probabilistically, and then adds that entry onto the word. if the index is zero, i.e.
meaning that you've reached the period character, then you end and print the word. done.
"""

# question: what does N[ix] concretely look like? is it a row vector or column vector?
# question: regardless of whether it's a row or column vector, shouldn't the ix = torch.multinomial(...)
# command produce a tuple of characters, not just one character? because each entry of the array
# is a pair of characters. but then the later code wouldn't make sense, and the statement itself wouldn't
# make sense either.




