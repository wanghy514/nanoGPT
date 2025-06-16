import os
import torch
import numpy as np
from time import time

from model import GPT, GPTConfig


def get_batch(data_dir, split, block_size, batch_size, device_type, device):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def load_model(init_from, device, out_dir=None, compile=False):
    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
        checkpoint = None

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    return model, checkpoint


def estimate_loss_wrapper(model, ctx, data_dir, split, num_batches, batch_size, attenuation_factor, method, device, device_type):

    tic = time()
    model.eval()    
    block_size = model.config.block_size
    losses = torch.zeros(num_batches)
    for k in range(num_batches):            
        X, Y = get_batch(data_dir, split, block_size, batch_size, device_type, device)
        with ctx:
            if method == "double_forward":            
                _, loss = model.double_forward_with_strongly_causal_attention(X, Y, attenuation_factor)
            elif method == "stepwise_forward":
                _, loss = model.stepwise_forward_with_strongly_causal_attention(X, Y, attenuation_factor)
        losses[k] = loss.item()
    print ("Evaluted %d batches with size %d, elappsed time= %.2f seconds" % (num_batches, batch_size, time()-tic))
    return losses.mean()
    