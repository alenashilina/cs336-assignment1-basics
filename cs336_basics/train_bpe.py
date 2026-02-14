import os
import json
import regex as re
from multiprocessing import Pool
from typing import BinaryIO

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


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def pre_tokenization(chunk: str, special_token: str, pre_tokenized_vocab: dict):
  splits = chunk.split(special_token)

  for seq in splits:
    for f in re.finditer(PAT, seq):
      key = tuple(bytes([b]) for b in seq[f.start(): f.end()].encode('utf-8'))
      pre_tokenized_vocab[key] = pre_tokenized_vocab.get(key, 0) + 1


def process_one_chunk(args):
    start, end, filepath = args
    local_vocab = {}

    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        pre_tokenization(chunk, "<|endoftext|>", local_vocab)

    return local_vocab


def merge_pair_in_token(pre_token, pair_to_merge, merged_bytes):
    new_token = []
    i = 0
    
    while i < len(pre_token):
        # Check if current position matches the pair to merge
        if i < len(pre_token) - 1 and (pre_token[i], pre_token[i+1]) == pair_to_merge:
            new_token.append(merged_bytes)
            i += 2  # Skip both elements of the pair
        else:
            new_token.append(pre_token[i])
            i += 1
    
    return tuple(new_token)


def train_tokenizer(vocab_size: int, datapath: str, special_token: list[str]):
    with open(datapath, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, special_token[0].encode('utf-8'))

    chunk_args = [(start, end, datapath) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(4) as pool:
        results = pool.map(process_one_chunk, chunk_args)
    
    # initializing the vocab
    vocab = {}
    vocab[0] = special_token[0].encode('utf-8')
    for i in range(256):
        vocab[i+1] = bytes([i])

    # Now merge all the local vocabs
    pre_tokenized_vocab = {}
    for local_vocab in results:
        for key, count in local_vocab.items():
            pre_tokenized_vocab[key] = pre_tokenized_vocab.get(key, 0) + count
    
    merges = []

    for j in range(vocab_size - 257):
        pair_counts = {}
        for pre_token, count in pre_tokenized_vocab.items():
            # For each adjacent pair in this pre_token
            for i in range(len(pre_token) - 1):
                pair = (pre_token[i], pre_token[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + count

        max_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        pair_to_merge, freq = max_pair

        merges.append(pair_to_merge)

        # Merge the two bytes into one
        merged_bytes = pair_to_merge[0] + pair_to_merge[1]
        vocab[len(vocab)] = merged_bytes  # Add at next index

        new_pre_tokenized_vocab = {}
        for pre_token, count in pre_tokenized_vocab.items():
            new_token = merge_pair_in_token(pre_token, pair_to_merge, merged_bytes)
            new_pre_tokenized_vocab[new_token] = count

        pre_tokenized_vocab = new_pre_tokenized_vocab
        if j % 1000 == 0:
          print(j)

    return vocab, merges


def save_tokenizer(vocab, merges, vocab_path="vocab.json", merges_path="merges.json"):
    # Convert vocab: bytes objects need to be encoded for JSON
    vocab_json = {k: v.decode('latin-1') for k, v in vocab.items()}
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab_json, f)
    
    # Convert merges: tuples of bytes to lists of strings
    merges_json = [[p[0].decode('latin-1'), p[1].decode('latin-1')] for p in merges]
    
    with open(merges_path, 'w') as f:
        json.dump(merges_json, f)
