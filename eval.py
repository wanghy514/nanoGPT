"""
Evaluate a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import numpy as np
from model import GPTConfig, GPT

from util import load_model, estimate_loss_wrapper

# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = None # "out-shakespeare-char"
dataset = ['shakespeare_char', 'openwebtext'][1]
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

model, checkpoint = load_model(init_from, device, out_dir=out_dir, compile=compile)
block_size = model.config.block_size
# poor man's data loader
data_dir = os.path.join('data', dataset)

# model.config.apply_sca_to_one_head = apply_sca_to_one_head
# l = estimate_loss_wrapper(
#     model,
#     ctx,
#     data_dir = data_dir,
#     split = 'val',
#     num_batches = num_batches,
#     batch_size = batch_size,
#     attenuation_factor = 1e10,
#     method = method,
#     device = device,
#     device_type = device_type,
# )
# print ('l=', l)

##### Choices #########
method_choices = ["double_forward"] #, "stepwise_forward"]
apply_sca_to_one_head_choices = [True, False]
attenution_factors = []
for e in range(1, 11):
    attenution_factors.extend([10**e, 2 * 10**e, 5 * 10 ** e])
#######################

num_batches = 10
batch_size = 8

val_losses = {}

for method in method_choices:
    for apply_sca_to_one_head in apply_sca_to_one_head_choices:
        key = f"using {method}, apply_sca_to_one_head={apply_sca_to_one_head}"
        val_losses[key] = []
        for f in attenution_factors:
            print (f"{key}, evaluting attenuation_factor={f}")
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
            val_losses[key].append(l.item())            

import pickle
with open("./results.pkl", "wb") as fp:
    pickle.dump((attenution_factors, val_losses), fp)

