import os
import torch
from contextlib import nullcontext
# from model import GPT, GPTConfig
from util import get_batch, load_model

# -----------------------------------------------------------------------------
out_dir = "out-shakespeare-char"
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
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

dataset = 'shakespeare_char'
data_dir = os.path.join('data', dataset)
block_size = 64
batch_size = 4

model, checkpoint = load_model("resume", device, out_dir=out_dir, compile=False)


def test_forward_batching_helper(fwd_func):
    X, Y = get_batch(data_dir, "val", block_size, batch_size, device_type, device)
    with ctx:
        logits, _ = fwd_func(X, Y)
        for i in range(batch_size):
            logits1, _ = fwd_func(X[i:i+1], Y[i:i+1])            
            diff = (logits1 - logits[i:i+1]).abs().mean() 
            # print ("diff=", diff)
            assert diff.item() < 1e-6
    print ("PASS")            


def test_forward_batching():
    test_forward_batching_helper(model.forward)

def test_stepwise_forward_batching():
    test_forward_batching_helper(
        lambda x, y : model.stepwise_forward_with_strongly_causal_attention(x, y, attenuation_factor=2e3)
    )

def test_double_forward_batching():
    test_forward_batching_helper(
        lambda x, y : model.double_forward_with_strongly_causal_attention(x, y, attenuation_factor=2e3)        
    )


if __name__ == "__main__":
    test_forward_batching()
    test_double_forward_batching()
    test_stepwise_forward_batching()
