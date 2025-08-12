import os
import json
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import regex as re
import requests

import torcetti

ENCODER_REMOTE_URL = (
    "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json"
)
VOCAB_REMOTE_URL = (
    "https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe"
)
CACHE_SUBDIR = os.path.join(".cache", "mingpt")

def bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs_chars = [chr(n) for n in cs]
    return dict(zip(bs, cs_chars))

def get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    if len(word) < 2:
        return set()
    pairs: Set[Tuple[str, str]] = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:

    def __init__(self, encoder: Dict[str, int], bpe_merges: List[Tuple[str, str]]):
        self.byte_encoder: Dict[int, str] = bytes_to_unicode()
        self.byte_decoder: Dict[str, int] = {v: k for k, v in self.byte_encoder.items()}
        self.encoder: Dict[str, int] = encoder
        self.decoder: Dict[int, str] = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks: Dict[Tuple[str, str], int] = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.cache: Dict[str, str] = {}

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word: Tuple[str, ...] = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram

            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)

        merged = " ".join(word)
        self.cache[token] = merged
        return merged

    def encode(self, text: str) -> List[int]:
        bpe_idx: List[int] = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode("utf-8")
            token_translated = "".join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(" ")
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
        return bpe_idx

    def decode(self, bpe_idx: Iterable[int]) -> str:
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        tokens_flat = "".join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        return tokens_bytes.decode("utf-8", errors="replace")

def get_file(local_file: str, remote_file: str, timeout: float = 15.0) -> None:
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file, timeout=timeout)
        response.raise_for_status()
        with open(local_file, "wb") as f:
            f.write(response.content)

def get_encoder(cache_dir: Optional[str] = None) -> Encoder:
    if cache_dir is None:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, CACHE_SUBDIR)
    os.makedirs(cache_dir, exist_ok=True)

    encoder_local_file = os.path.join(cache_dir, "encoder.json")
    get_file(encoder_local_file, ENCODER_REMOTE_URL)
    with open(encoder_local_file, "r") as f:
        encoder: Dict[str, int] = json.load(f)
    assert len(encoder) == 50257

    vocab_local_file = os.path.join(cache_dir, "vocab.bpe")
    get_file(vocab_local_file, VOCAB_REMOTE_URL)
    with open(vocab_local_file, "r", encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges: List[Tuple[str, str]] = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]
    assert len(bpe_merges) == 50000

    return Encoder(encoder, bpe_merges)

class BPETokenizer:

    def __init__(self) -> None:
        self.encoder = get_encoder()

    def __call__(self, text: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        assert isinstance(text, str)
        idx = [self.encoder.encode(text)]
        return torcetti.tensor(idx, dtype=np.int64)

    def decode(self, idx) -> str:
        assert len(idx.shape) == 1
        return self.encoder.decode(idx.tolist())


if __name__ == '__main__':

    e = get_encoder()
    r = e.encode_and_show_work(text)