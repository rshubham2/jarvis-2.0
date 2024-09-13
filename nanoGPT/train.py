# train.py

import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2LMHeadModel, GPT2Model
import pandas as pd
import joblib

# -----------------------------------------------------------------------------
# Configuration
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'gpt2'  # 'scratch' or 'resume' or 'gpt2*'

# WandB logging
wandb_log = False  # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())

# Data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

backend = 'nccl'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16'
compile = False
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# Add any command-line overrides or config file overrides here
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# Initialize distributed data parallel if needed
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single GPU and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Load dataset
data_dir = os.path.join('data', dataset)
def get_batch(split):
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

# Initialize iteration counters
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")
else:
    print("No meta.pkl found, using default vocab_size of 50257")
    meta_vocab_size = 50257  # Default GPT-2 vocab size

# Model initialization
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, vocab_size=meta_vocab_size, dropout=dropout)
if init_from == 'scratch':
    # Initialize a new model from scratch
    print("Initializing a new model from scratch")
    model = GPT2LMHeadModel(config=GPT2Config(**model_args))
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # Resume training from a checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT2LMHeadModel(config=GPT2Config(**checkpoint['model_args']))
    model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # Initialize from OpenAI GPT-2 weights
    model = GPT2LMHeadModel.from_pretrained(init_from)
    tokenizer = GPT2Tokenizer.from_pretrained(init_from)
    # Update model_args with the loaded model's config
    model_args.update({
        'n_layer': model.config.n_layer,
        'n_head': model.config.n_head,
        'n_embd': model.config.n_embd,
        'vocab_size': model.config.vocab_size,
    })
else:
    raise ValueError(f"Unknown init_from value: {init_from}")

# Move model to device
model.to(device)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
if init_from == 'resume' and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])

# Compile the model if needed
if compile:
    print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model)  # Requires PyTorch 2.0

# Initialize GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# DDP wrapper if needed
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# Evaluation function
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                outputs = model(X, labels=Y)
                loss = outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Learning rate scheduler
def get_lr(it):
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # Cosine decay
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Load emotions and datasets
def load_emotions():
    # Load or define your emotion labels here
    return ['anger', 'joy', 'sadness', 'fear', 'surprise', 'disgust']

def load_dataset():
    # Load your dataset of situations and responses
    data = pd.read_csv('enhanced_empathy_responses.csv')  # Replace with your dataset path
    situations = data['situation'].tolist()
    responses = data['response'].tolist()
    return situations, responses
# Load models and tokenizers
def load_models_and_tokenizers():
    # Load pre-trained models and tokenizers
    emotion_model = GPT2ForSequenceClassification.from_pretrained('gpt2')
    emotion_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    response_model = GPT2LMHeadModel.from_pretrained('gpt2')
    response_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    similarity_model = GPT2Model.from_pretrained('gpt2')
    return emotion_model, emotion_tokenizer, response_model, response_tokenizer, similarity_model

# Predict emotion function
def predict_emotion(emotion_model, emotion_tokenizer, text, device, emotions):
    inputs = emotion_tokenizer(text, return_tensors='pt').to(device)
    outputs = emotion_model(**inputs)
    logits = outputs.logits.detach().cpu()
    probabilities = torch.softmax(logits, dim=-1).squeeze()
    confidence, prediction = torch.max(probabilities, dim=0)
    emotion = emotions[prediction.item()]
    return emotion, confidence.item()

# Retrieve similar example function
def retrieve_similar_example(user_input, situations, responses, similarity_model):
    # Simple similarity retrieval based on embeddings
    similarity_model.eval()
    with torch.no_grad():
        inputs = similarity_model.transformer.wte.weight.mean(dim=0).unsqueeze(0)  # Dummy embedding
        user_embedding = similarity_model(**similarity_model.transformer.wte.weight.mean(dim=0).unsqueeze(0))
        max_similarity = -1
        best_index = -1
        for idx, situation in enumerate(situations):
            situation_inputs = similarity_model.transformer.wte.weight.mean(dim=0).unsqueeze(0)  # Dummy embedding
            situation_embedding = similarity_model(**situation_inputs)
            similarity = torch.cosine_similarity(user_embedding.last_hidden_state.mean(dim=1),
                                                 situation_embedding.last_hidden_state.mean(dim=1))
            if similarity > max_similarity:
                max_similarity = similarity
                best_index = idx
    similar_situation = situations[best_index]
    retrieved_response = responses[best_index]
    similarity_score = max_similarity.item()
    return similar_situation, retrieved_response, similarity_score

# Generate response function
def generate_response_from_example(similar_situation, retrieved_response, response_model, response_tokenizer, user_input, device, emotion):
    prompt = f"User is feeling {emotion}. They said: '{user_input}'. Assistant should respond appropriately."
    inputs = response_tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = response_model.generate(inputs, max_length=150)
    response = response_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Handle user input function
def handle_user_input(emotion_model, emotion_tokenizer, response_model, response_tokenizer, similarity_model, emotions, device):
    while True:
        user_input = input("Enter your message (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        # Predict emotion
        emotion, confidence = predict_emotion(emotion_model, emotion_tokenizer, user_input, device, emotions)
        print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")

        # Generate response
        situations, responses = load_dataset()
        similar_situation, retrieved_response, similarity_score = retrieve_similar_example(user_input, situations, responses, similarity_model)
        response = generate_response_from_example(similar_situation, retrieved_response, response_model, response_tokenizer, user_input, device, emotion)

        print(f"Model response: {response}")

# Main training loop
if __name__ == "__main__":
    # Load models and tokenizers
    emotion_model, emotion_tokenizer, response_model, response_tokenizer, similarity_model = load_models_and_tokenizers()
    emotions = load_emotions()

    # Move models to device
    emotion_model.to(device)
    response_model.to(device)
    similarity_model.to(device)

    # Start the input handling loop
    handle_user_input(emotion_model, emotion_tokenizer, response_model, response_tokenizer, similarity_model, emotions, device)

    # Initialize variables for training loop
    X, Y = get_batch('train')  # Fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # Number of iterations in the lifetime of this process
    running_mfu = -1.0

    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                import wandb
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # Convert to percentage
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"Saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if iter_num == 0 and eval_only:
            break

        # Forward, backward, update
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                outputs = model(X, labels=Y)
                loss = outputs.loss / gradient_accumulation_steps
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # Backward pass with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # Clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # Step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # Flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # Let the training loop settle a bit
                # mfu calculation (model flops utilization)
                mfu = 0  # Placeholder, implement your own calculation
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"Iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()
