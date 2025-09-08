import os
from typing import BinaryIO, Iterable, List, Tuple
import regex as re
import multiprocessing

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

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

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


def pretokenize(chunk: str, special_tokens: list[str]) -> dict[str, int]:
    # Removing special tokens before pre-tokenization
    # This can be done using re.split with "|".join(special_tokens) as the delimiter 
    text_list = re.split("|".join(special_tokens), chunk)
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = {}
    for chunk in text_list:
        find_words = re.findall(pattern, chunk)
        for word in find_words:
            # word = word.strip()
            pre_tokens[word] = 1 + pre_tokens.get(word, 0)
    return pre_tokens


def find_all_occurrences(main_str, sub_str):
    indexes = []
    start = 0 
    sub_len = len(sub_str)
    main_len = len(main_str)
    
    if sub_len == 0 or sub_len > main_len:
        return indexes
    
    while True:
        pos = main_str.find(sub_str, start)
        if pos == -1:  
            break
        indexes.append(pos)
        start = pos + 1
    return indexes       

def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
):

    """
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
    
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation.
    """
    vocab: dict[int, bytes] = {} # dict[int, bytes]
    merges: list[tuple[bytes, bytes]] = [] 

    token_id = 0
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1

    for token in special_tokens:
        vocab[token_id] = token.encode("utf-8")
        token_id += 1
    
    num_processes = os.cpu_count()

    pre_token_count: dict[str, int] = {}
    pre_token_list: list[str] = []

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        start_end_pairs = list(zip(boundaries[:-1], boundaries[1:]))
        chunk_list = []
        for start, end in start_end_pairs:
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8', errors='ignore')
            chunk_list.append(chunk)
        num_processes = min(num_processes, len(chunk_list))
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(pretokenize, [(chunk, special_tokens) for chunk in chunk_list])
            for res in results:
                for token, count in res.items():
                    if token not in pre_token_count:
                        pre_token_list.append(token)
                    pre_token_count[token] = pre_token_count.get(token, 0) + count

    pre_token_sequences = []
    for i in range(len(pre_token_list)):
        pre_token = pre_token_list[i]
        bytes_data = pre_token.encode('utf-8')
        sequence = [bytes([b]) for b in bytes_data]
        pre_token_sequences.append(sequence)
    

    token_pair_count: dict[tuple[bytes, bytes], int] = {}
    token_pair_positions: dict[tuple[bytes, bytes], set[int]] = {}

    # traverse pre_tokens and initialize token_pairs count
    for i in range(len(pre_token_list)):
        tokens = pre_token_sequences[i]
        pre_token = pre_token_list[i]
        count_i = pre_token_count[pre_token]
        for j in range(0, len(tokens)-1):
            token_pair = (tokens[j], tokens[j+1])
            token_pair_count[token_pair] = token_pair_count.get(token_pair, 0) + count_i
            if token_pair not in token_pair_positions:
                token_pair_positions[token_pair] = set()    
            token_pair_positions[token_pair].add(i)

    merge_count_list = []
    # train bpe tokenizer
    while len(vocab) < vocab_size :
        # 1. sort token_pair_count and merge the most frequent pair to a new token
        merge_pair_count = max(token_pair_count.items(), key = lambda item:(item[1], item[0]))  
        merge_pair = merge_pair_count[0]
        count = merge_pair_count[1]
        if count < 2:
            break
        merge_count_list.append(count)
        merges.append(merge_pair)
        new_token = merge_pair[0] + merge_pair[1]
        current_len = len(vocab)
        vocab[current_len] = new_token

        # 2. update token_pair_count and token_pair_positions
        merge_positions = token_pair_positions[merge_pair].copy()
        # del token_pair_count[merge_pair]
        # del token_pair_positions[merge_pair]

        for pos in merge_positions:
            origin_sequence = pre_token_sequences[pos]
            pre_token = pre_token_list[pos]
            pre_token_count_value = pre_token_count[pre_token]
            new_sequence = []
            i = 0
            while i < len(origin_sequence):
                if i<len(origin_sequence)-1 and origin_sequence[i]==merge_pair[0] and origin_sequence[i+1]==merge_pair[1]:
                    new_sequence.append(new_token)
                    i += 2
                else:
                    new_sequence.append(origin_sequence[i])
                    i += 1
            

            for j in range(0, len(origin_sequence) - 1):
                old_pair = (origin_sequence[j], origin_sequence[j+1])
                if old_pair in token_pair_count:
                    token_pair_count[old_pair] -= pre_token_count_value
                if token_pair_count[old_pair] <= 0:
                    del token_pair_count[old_pair]
                if old_pair in token_pair_positions:
                    token_pair_positions[old_pair].discard(pos)
                    if not token_pair_positions[old_pair]:
                        del token_pair_positions[old_pair]

            for j in range(0, len(new_sequence) - 1):
                new_pair = (new_sequence[j], new_sequence[j+1])
                if new_pair not in token_pair_count:
                    token_pair_count[new_pair] = 0
                token_pair_count[new_pair] += pre_token_count_value
                if new_pair not in token_pair_positions:
                    token_pair_positions[new_pair] = set()
                token_pair_positions[new_pair].add(pos)

            pre_token_sequences[pos] = new_sequence

        if merge_pair in token_pair_count:
            del token_pair_count[merge_pair]
        if merge_pair in token_pair_positions:
            del token_pair_positions[merge_pair]

    return vocab, merges
  


