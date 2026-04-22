import torch
import torch.nn as nn
from torch.nn import functional as f

torch.manual_seed(1337)

device = 'mps' if torch.backends.mps.is_available() else 'cpu' # mps : metal performance shaders
print(f"Using device: {device}")

with open('gita_clean.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Building Our Vocabulary ------

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

# Data Split -------

n = int(0.9*len(data)) # 90% of data
train_data = data[:n] # 90% train data set
val_data = data[n:] # 10% validation data set

print(f"Train Data: {len(train_data):,}, Validation Data: {len(val_data):,}")

# HYPERPARAMETERS -------

batch_size = 32

block_size = 8

max_iters = 3000

eval_iters = 200

learning_rate = 1e-2

eval_interval = 300

# Batch Loader --------

def get_batch(split):
    data_split = train_data if split == 'train' else val_data

    ix = torch.randint(len(data_split) - block_size, (batch_size,))

    x = torch.stack([data_split[i : i+block_size] for i in ix])
    y = torch.stack([data_split[i+1 : i+block_size+1] for i in ix])

    # x = [20, 30, 40]
    # y = [30, 40, 50]

    return x.to(device), y.to(device)

# Loss Estimator ---------

@torch.no_grad() # @torch.no_grad() tells PyTorch: "don't track gradients here". We're only MEASURING loss, not updating weights — so no need to compute gradients. This saves memory and speeds things up.
def estimate_loss():
    out = {}
    model_eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in eval_iters:
            X,Y = get_batch(split)
            _, loss = model(X,Y)
            losses[k] = loss.items()
            out[split] = losses.mean()
    model.train()
    return out