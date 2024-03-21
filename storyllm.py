import torch
import torch.nn as nn

from torch.nn import functional as F

# Hyperparameters
batch_size = 64 # Number of independent sequences to process in parallel
block_size = 256 # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384 # number of embedding dimensions
n_layer = 6 # number of transformer layers
n_head = 6 # number of heads in multi-head attention
dropout = 0.2 # dropout rate

# Set the device to use cuda if there's a GPU.  
# Note that later we move the model and data to the device (by passing device in args)
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

# First lets get the tiny shakespeare dataset - uncomment to download again!
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Load the data
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Find the unique characters that occur in the text
chars = sorted(list(set(text)))
vocabulary_size = len(chars)

# Create very simple tokenizer - maps characters to integers and vice versa
# Look at the other tokenizers that are used.
# Google uses SentencePiece.  OpenAI uses tiktoken.
str_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_str = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [str_to_int[ch] for ch in s] # encoder - converts string to integers
decode = lambda x: ''.join([int_to_str[i] for i in x]) # decoder - converts integers to string

# Encode the entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Create train and test data sets
# We will use 90% of the data for training and 10% for validation (change n for different percentages)
n = int(0.9*len(data))
train_data = data[:n]
validation_data = data[n:]

# Data loading
def get_batch(split):
    # Generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() # Disables Backward Propagation for more efficient pytorch memory use
def estimate_loss():
    out = {}
    model.eval() # Set model for evaluation phase
    for split in ['train', 'validation']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[i] = loss.item() # get loss
        out[split] = losses.mean() # get avg loss over both splits (a lot less noisy)
    model.train() # Set model back to training phase
    return out

class Head(nn.Module):
    """Head for the transformer model - one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # tril = buffer not parameter

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute the self-attention
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (T, T)
        weights = F.softmax(weights, dim=-1) # (T, T) # exponentiate and normalize
        weights = self.dropout(weights) # (T, T) # randomly prevent some of the nodes from communicating

        # perform weighted aggregation of values
        v = self.value(x) # (B, T, C)
        out = weights @ v 
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple Heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply each head to the input in parallel, concatenate outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    

class FeedForward(nn.Module):
    """ Simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 4* comes from paper 2048/512 = 4 (see 1:32 in the video)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # residual layer going back into projection pathway
            nn.Dropout(dropout),  # dropout layer before residual connection
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer Block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head = number of heads we would like to use
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# GPT Model
class GPTLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()

        # Each token directly reads off the logits for the next token from a lookup table
        # has a weight inside that stores the probability of the next token
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocabulary_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):

        B, T = idx.shape

        # idx and targers are both (B,T) tensor of integers
        #logits = self.token_embedding_table(idx) # (B, T, C)
        tok_emb = self.token_embedding_table(idx) # (B, T, C) # instead of tokens get token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)

        # To go from token embeddings to logits, need a linear layer
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # reshape logits so we can use cross-entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)  # -1 means "infer this dimension" (translates to B*T)

            # evaluate the loss function (quality of predictions)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
    # idx is a (B, T) array of indices in the current 
    # the job of generate is to take a BxT and return a BxT+1, +2, +3, etc

        for _ in range(max_new_tokens):
            # crop context so it never exceeds block size
            idx_cond = idx[:, -block_size:]

            # Get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)    # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
    
# Create the model
model = GPTLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    # Every once in a while evaluate the loss on the training and validation data
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f'Iter {iter} - Train Loss: {losses["train"]:.4f}, Validation Loss: {losses["validation"]:.4f}')

    # Get a batch of data
    xb, yb = get_batch("train")

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device) # creating a 1x1 tensor of zeros (remember 0 = new line)

# ask for 500 new tokens, generate, convert to list to feed into decode
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))

# Save the output in a file
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))