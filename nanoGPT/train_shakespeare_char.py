out_dir = 'out-shakespeare-char'
log_interval = 2 
always_save_checkpoint = True
wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1

  
lr_decay_iters = 1000  
min_lr = 1e-5  
beta2 = 0.98   

warmup_iters = 200  # Increase warmup to help with stability in the initial phase

batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 300
learning_rate = 1e-2
device = 'cpu'
eval_iters = 200