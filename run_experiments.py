import os

# base_cmd = "python train.py config/train_shakespeare_char.py" + \
#     " --device=cpu --compile=False --eval_iters=20 --log_interval=50" + \
#     " --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128" + \
#     " --max_iters=2000 --lr_decay_iters=2000 --eval_interval=200 --dropout=0.0"


base_cmd = "python train.py config/train_gpt2.py --device=cpu --compile=False"

# fast debugging 
base_cmd += " --max_iters=3 --eval_interval=1 --eval_iters=2 --log_interval=1"


out_dirs = []
out_dir = "baseline"
out_dirs.append(out_dir)
cmd = base_cmd + " --use_ar=False"
os.system(cmd)

for ar_head_choice in ["ALL", "FIRST"]:
    for ar_attenuation in [1e4, 1e5, 1e6]:

        out_dir = f"exp_head_{ar_head_choice}_attenuation_{str(ar_attenuation)}"
        out_dirs.append(out_dir)
        
        cmd = base_cmd + \
            f" --use_ar=True" + \
            f" --ar_head_choice={ar_head_choice}" + \
            f" --out_dir={out_dir}" + \
            f" --ar_attenuation={ar_attenuation}"    
        
        os.system(cmd)

print ("======================================")
print ("out_dirs=", out_dirs)
print ("======================================")
