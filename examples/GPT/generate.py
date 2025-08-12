import torcetti

from examples.GPT.hf_compat import HFGPT
from examples.GPT.gpt_model import GPT
from examples.GPT.bpe import BPETokenizer

from transformers import GPT2Tokenizer, GPT2LMHeadModel


use_mingpt = True
model_type = 'gpt2'

if use_mingpt:
    model = GPT(vocab_size=50257, embed_dim=768, num_heads=12, num_layers=12, max_seq_len=1024, dropout=0.1)
else:
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.config.pad_token_id = model.config.eos_token_id

model.eval()

def sample(model, x, max_new_tokens=20, do_sample=True, top_k=40):
    for i in range(max_new_tokens):
        logits = model(x)
        logits = logits[:, -1, :]
        if do_sample:
            probs = torcetti.nn.functional.softmax(logits, axis=-1)
            
            if top_k > 0 and top_k < probs.shape[-1]:
                top_k_probs, top_k_indices = torcetti.topk(probs, k=top_k, dim=-1)
                
                mask = torcetti.zeros_like(probs)
                
                for b in range(probs.shape[0]):
                    for k in range(top_k):
                        mask.data[b, top_k_indices.data[b, k]] = 1
                
                probs = probs * mask
                probs = probs / probs.sum(axis=-1, keepdims=True)
            
            next_token = torcetti.multinomial(probs, num_samples=1)
        else:
            next_token = torcetti.argmax(logits, axis=-1, keepdims=True)
        x = torcetti.cat([x, next_token], axis=1)
    return x

def generate(prompt='', num_samples=10, steps=20, do_sample=True):
        
    if use_mingpt:
        tokenizer = BPETokenizer()
        if prompt == '':
            x = torcetti.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torcetti.core.dtype.get_default_dtype())
        else:
            x = tokenizer(prompt)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        if prompt == '': 
            prompt = '<|endoftext|>'
        encoded_input = tokenizer(prompt, return_tensors='pt')
        x = torcetti.tensor(encoded_input['input_ids'].numpy(), dtype=torcetti.core.dtype.get_default_dtype())
    
    x = x.expand(num_samples, -1)

    y = sample(model, x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    for i in range(num_samples):
        token_ids = y[i].data
        out = tokenizer.decode(token_ids)
        print('-'*80)
        print(out)

generate(prompt='Once upon a time', num_samples=3, steps=20)
