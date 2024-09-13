import os
import pickle
import requests
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm 

# Load the tokenizer globally
enc = tiktoken.get_encoding("gpt2")

# Load the openwebtext-10k dataset
dataset = load_dataset("stas/openwebtext-10k")

# Define the process function
def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    return {'ids': ids, 'len': len(ids)}

# Split the dataset into train and validation sets
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

# Tokenize and process the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=4,
)

# Get vocabulary information
vocab_size = enc.n_vocab
print(f"Vocab size: {vocab_size}")

# Save the encoded train and validation data to binary files
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    idx = 0
    total_batches = min(1024, len(dset))

    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        arr[idx:idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
        arr.flush()

# Save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': {i: chr(i) for i in range(vocab_size)},
    'stoi': {chr(i): i for i in range(vocab_size)},
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)