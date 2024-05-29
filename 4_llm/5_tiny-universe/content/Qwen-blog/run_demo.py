from transformers_441.models.qwen2 import Qwen2Config, Qwen2Model
import torch
import time

def run_qwen2():
    qwen2config = Qwen2Config(vocab_size=151936,
                              hidden_size=4096//2,
                              intermediate_size=22016//2,
                              num_hidden_layers=32//2,
                              num_attention_heads=32,  # 每一头的hidden_dim=2048/32=64
                              max_position_embeddings=2048//2,
                              _attn_implementation='eager')
    qwen2model = Qwen2Model(config=qwen2config)

    input_ids = torch.randint(0, qwen2config.vocab_size, (4,30))

    res = qwen2model(input_ids)
    print(type(res))

if __name__ == "__main__":
    t1 = time.time()
    run_qwen2()
    print(time.time()-t1)