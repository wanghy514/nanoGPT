"""
Evaluate a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = "out-shakespeare-char"
dataset = 'shakespeare_char'
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

from util import load_model, estimate_loss_wrapper
model, checkpoint = load_model(init_from, device, out_dir=out_dir, compile=compile)

"""
Model trained by:
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
"""
import numpy as np
block_size = model.config.block_size


# poor man's data loader
data_dir = os.path.join('data', dataset)

method = ["double_forward", "stepwise_forward"][1]


attenution_factors = []
for e in range(1, 11):
    attenution_factors.extend([10**e, 2 * 10**e, 5 * 10 ** e])
num_batches = 10
batch_size = 8

val_losses = []
for f in attenution_factors:
    print ("Evaluting f=", f)
    l = estimate_loss_wrapper(
        model,
        ctx,
        data_dir = data_dir,
        split = 'val',
        num_batches = num_batches,
        batch_size = batch_size,
        attenuation_factor = f,
        method = method,
        device = device,
        device_type = device_type,
    )
    val_losses.append(l.item())

print ("attenution_factors=", attenution_factors)
print ("val_losses=", val_losses)
