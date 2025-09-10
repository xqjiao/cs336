import pickle
import regex as re
from typing import Iterable, Iterator
class BPE_Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab # id to token mapping
        self.merges = merges # list[tuple(bytes, bytes)]
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.token_to_id = {token: idx for idx, token in vocab.items()}
        self.merges_rank = {merge: i for i, merge in enumerate(merges)}

        
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'rb') as vf:
            vocab = pickle.load(vf)
        with open(merges_filepath, 'rb') as mf:
            merges = pickle.load(mf)
        return cls(vocab, merges, special_tokens)



    def encode(self, text) -> list[int]:
        if self.special_tokens:
            pattern = "|".join([re.escape(token) for token in self.special_tokens])
            pattern = f"({pattern})"
            text_list = re.split(pattern, text)
        else:
            text_list = [text]
        result_ids = []
        chunk_count = len(text_list)
        # print(f"Number of chunks after splitting by special tokens: {chunk_count}")
        for i, chunk in enumerate(text_list):
            if chunk in self.special_tokens:
                token = chunk.encode('utf-8')
                if token in self.token_to_id:
                    result_ids.append(self.token_to_id[token])
                continue
            pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pre_tokens = re.findall(pattern, chunk)
            for pre_token in pre_tokens:
                tokens = pre_token.encode('utf-8')
                bytes_sequence = [bytes([b]) for b in tokens]
                while len(bytes_sequence) > 1:
                    merge_pair = None
                    min_rank = float('inf')
                    pairs = [(bytes_sequence[i], bytes_sequence[i+1]) for i in range(len(bytes_sequence)-1)]
                    for pair in pairs:
                        if pair in self.merges:
                            rank = self.merges_rank[pair]
                            if rank < min_rank:
                                min_rank = rank
                                merge_pair = pair
                    if merge_pair is None:
                        break
                    new_sequence = []
                    i = 0
                    while i < len(bytes_sequence):
                        if i < len(bytes_sequence) - 1 and (bytes_sequence[i], bytes_sequence[i+1]) == merge_pair:
                            new_token = merge_pair[0] + merge_pair[1]
                            new_sequence.append(new_token)
                            i += 2
                        else:
                            new_sequence.append(bytes_sequence[i])
                            i += 1
                    bytes_sequence = new_sequence

                for token in bytes_sequence:
                    if token in self.token_to_id:
                        result_ids.append(self.token_to_id[token])
                    else:
                        raise ValueError(f"Token {token} not in vocabulary.")
        return result_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        result  = b''
        for idx in ids:
            if idx in self.vocab:
                result += self.vocab[idx]
            else:
                raise ValueError(f"Token ID {idx} not in vocabulary.")
        result = result.decode('utf-8', errors='replace')
        return result

