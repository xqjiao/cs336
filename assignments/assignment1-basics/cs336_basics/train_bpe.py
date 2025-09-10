import os
import time
import json
import threading
import psutil
from bpe_tokenizer import train_bpe_tokenizer
import pickle


# 监控内存峰值
max_memory = 0
monitoring = False

def monitor_memory():
    global max_memory, monitoring
    process = psutil.Process(os.getpid())
    while monitoring:
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        if current_memory > max_memory:
            max_memory = current_memory
        time.sleep(0.01)


def train_bpe_tokenizer_with_monitoring(dataset_path, vocab_size,special_tokens):
    # 开始内存监控
    global monitoring
    monitoring = True
    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.start()
    dataset_name = os.path.basename(dataset_path)
    dataset_name = dataset_name.split('.')[0]
    # 训练tokenizer
    print("Training tokenizer on dataset:", dataset_name)
    print("Vocabulary size:", vocab_size)
    start_time = time.time()
    vocab, merges = train_bpe_tokenizer(dataset_path, vocab_size,special_tokens)
    end_time = time.time()

    # 停止内存监控
    monitoring = False
    monitor_thread.join()

    training_time = end_time - start_time
    memory_used = max_memory

    print("Training completed.")

    save_dir = "/data/xqjiao/cs336/assignments/assignment1-basics/output/bpe_tokenizers"
    save_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # 保存词汇表和合并列表, 词汇表json格式，合并列表txt格式

    with open(os.path.join(save_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(save_dir, "merges.pkl"), "wb") as f:
        pickle.dump(merges, f)

    vocab_str = {key: value.decode('utf-8', errors='ignore') for key, value in vocab.items()}
    with open(os.path.join(save_dir, "vocab.json"), 'w', encoding='utf-8') as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=4)

    with open(os.path.join(save_dir, "merges.txt"), 'w', encoding='utf-8') as f:
        for i,merge in enumerate(merges):
            merge_str = f"({merge[0].decode('utf-8', errors='ignore')}, {merge[1].decode('utf-8', errors='ignore')}) -- {i}"
            f.write(merge_str + "\n")

    # 找到最长的token
    longest_token = ""
    for token in vocab.values():
        if len(token) > len(longest_token):
            longest_token = token

    longest_token_length = len(longest_token)

    # 输出结果
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Training time: {training_time / 60:.2f} minutes")
    print(f"Peak memory usage: {memory_used:.2f} MB")
    print(f"Longest token: {repr(longest_token)} with length {longest_token_length}")

if __name__ == "__main__":
    dataset_path = "/data/xqjiao/cs336/assignments/assignment1-basics/data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    train_bpe_tokenizer_with_monitoring(dataset_path, vocab_size,special_tokens)
    # dataset_path = "/data/xqjiao/cs336/assignments/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10000
    # special_tokens = ["<|endoftext|>"]
    # train_bpe_tokenizer_with_monitoring(dataset_path, vocab_size,special_tokens)