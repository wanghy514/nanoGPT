import os

out_dirs = []
for ar_attenuation in [1e4, 1e5, 1e6]:

    out_dir = f"exp_attenuation_{str(ar_attenuation)}"
    out_dirs.append(out_dir)

    cmd = "python train.py config/train_shakespeare_char.py" + \
        " --device=cpu --compile=False --eval_iters=20 --log_interval=50" + \
        " --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128" + \
        " --max_iters=2000 --lr_decay_iters=2000 --eval_interval=200 --dropout=0.0" + \
        f" --out_dir={out_dir}" + \
        f" --ar_attenuation={ar_attenuation}"    
    
    os.system(cmd)

print ("======================================")
print ("out_dirs=", out_dirs)
print ("======================================")
