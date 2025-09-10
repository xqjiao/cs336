from typing import BinaryIO, Iterable, List, Tuple
import multiprocessing
import os
import numpy as np
from tokenizer_class import BPE_Tokenizer
import regex as re

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]

    # chunk_count = len(chunk_boundaries) - 1
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096*2  # Read ahead by 4k*2 bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks

    return sorted(set(chunk_boundaries))

def iterate_encoding(tokenizer, chunk: str, special_tokens: list[str]) -> List[int]:
    text_list = re.split("|".join(special_tokens), chunk)
    res = list(tokenizer.encode_iterable(text_list))
    return res



def encode_dataset(dataset_path, tokenizer, output_path, special_tokens):
    if os.path.exists(output_path):
        print(f"{output_path} already exists, skipping encoding.")
        return
    
    dataset_name = dataset_path.split('/')[-1].split('.')[0]
    print(f"Encoding {dataset_name} dataset...")
    output = []
    num_workers = os.cpu_count() or 1
    print(f"Using {num_workers} workers for encoding.")
    with open(dataset_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_workers, split_special_token=b"<|endoftext|>")
        start_end_pairs = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
        print(f"Dataset chunked into {len(start_end_pairs)} parts.")    
        chunk_list = []
        for start, end in start_end_pairs:
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            chunk_list.append(chunk)
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        results = pool.starmap(iterate_encoding, [(tokenizer, chunk, special_tokens) for chunk in chunk_list])
    for res in results:
        output.extend(res)
    
    
    print(f"{dataset_name} dataset encoded, total length: {len(output)}")
    print(f"saved to {output_path}")
    np.save(output_path, np.array(output, dtype=np.uint16))



print("Loading tokenizers...")
ts_dataset_path = "/data/xqjiao/cs336/assignments/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
vocab_path = "/data/xqjiao/cs336/assignments/assignment1-basics/output/bpe_tokenizers/TinyStoriesV2-GPT4-train/vocab.pkl"
merges_path = "/data/xqjiao/cs336/assignments/assignment1-basics/output/bpe_tokenizers/TinyStoriesV2-GPT4-train/merges.pkl"
special_tokens = ['<|endoftext|>']
ts_tokenizer = BPE_Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens)

owt_dataset_path = "/data/xqjiao/cs336/assignments/assignment1-basics/data/owt_valid.txt"
vocab_path = "/data/xqjiao/cs336/assignments/assignment1-basics/output/bpe_tokenizers/owt_train/vocab.pkl"
merges_path = "/data/xqjiao/cs336/assignments/assignment1-basics/output/bpe_tokenizers/owt_train/merges.pkl"
special_tokens = ['<|endoftext|>']
owt_tokenizer = BPE_Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath=merges_path, special_tokens=special_tokens)
print("Tokenizers loaded.")


ts_train_dataset_path = "/data/xqjiao/cs336/assignments/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
ts_valid_dataset_path = "/data/xqjiao/cs336/assignments/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
owt_train_dataset_path = "/data/xqjiao/cs336/assignments/assignment1-basics/data/owt_train.txt"
owt_valid_dataset_path = "/data/xqjiao/cs336/assignments/assignment1-basics/data/owt_valid.txt"

special_tokens = ['<|endoftext|>']

output_dir = "/data/xqjiao/cs336/assignments/assignment1-basics/output/encoded_datasets"
os.makedirs(output_dir, exist_ok=True)

test_dataset_path = "/data/xqjiao/cs336/assignments/assignment1-basics/tests/fixtures/tinystories_sample.txt"
encode_dataset(test_dataset_path, ts_tokenizer, os.path.join(output_dir, "test.npy"), special_tokens)
encode_dataset(ts_valid_dataset_path, ts_tokenizer, os.path.join(output_dir, "ts_valid.npy"), special_tokens)
encode_dataset(ts_train_dataset_path, ts_tokenizer, os.path.join(output_dir, "ts_train.npy"), special_tokens)
encode_dataset(owt_valid_dataset_path, owt_tokenizer, os.path.join(output_dir, "owt_valid.npy"), special_tokens)
encode_dataset(owt_train_dataset_path, owt_tokenizer, os.path.join(output_dir, "owt_train.npy"), special_tokens)
