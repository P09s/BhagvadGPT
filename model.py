import torch
import torch.nn as nn
from torch.nn import functional as f

torch.manual_seed(1337)

device = 'mps' if torch.backends.mps.is_available() else 'cpu' # mps : metal performance shaders
print(f"Using device: {device}")

with open('gita_clean.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Building Our Vocabulary

# after data cleaning and importing that clean data file here
# now to give this data to a NN we need to convert these characters into unique integers 

chars = sorted(set(text))
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {''.join(chars)}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l]) 

# Convert whole text into a tensor 
data = torch.tensor(encode(text), dtype=torch.long)
# data is now a 1D tensor of 142,212 integers like [45, 12, 67, 3, ...]