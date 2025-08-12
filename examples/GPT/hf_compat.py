from __future__ import annotations

from typing import Any


from examples.GPT.gpt_model import GPT

class HFGPT(GPT):
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "HFGPT":
        from transformers import GPT2LMHeadModel 

        hf_model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        cfg = hf_model.config

        embed_dim = int(cfg.n_embd)
        num_heads = int(cfg.n_head)
        num_layers = int(cfg.n_layer)
        vocab_size = int(cfg.vocab_size)
        max_seq_len = int(getattr(cfg, "n_positions", getattr(cfg, "n_ctx", 1024)))

        model = cls(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=0.0,
        )
        model.load_from_hf_gpt2(hf_model)
        return model

    def load_from_hf_gpt2(self, hf_model: Any) -> None:

        import numpy as np
        
        # Ensure both models are in eval mode
        hf_model.eval()
        self.eval()
        
        # Helper function to convert HF tensors to numpy with matching dtype
        to_np = lambda t: t.detach().cpu().numpy().astype(self.lm_head.weight.data.dtype)

        def _assign_linear(linear, W_np, b_np):
            import numpy as _np
            target_w_shape = linear.weight.data.shape
            if W_np.shape == target_w_shape:
                linear.weight.data[...] = W_np
            elif W_np.T.shape == target_w_shape:
                linear.weight.data[...] = W_np.T
            else:
                raise ValueError(f"Weight shape mismatch: got {W_np.shape}, expected {target_w_shape}")

            if linear.bias is not None:
                if b_np is None:
                    linear.bias.data[...] = _np.zeros_like(linear.bias.data)
                else:
                    if b_np.shape != linear.bias.data.shape:
                        raise ValueError(f"Bias shape mismatch: got {b_np.shape}, expected {linear.bias.data.shape}")
                    linear.bias.data[...] = b_np
        
        self.transformer.wte.weight.data[...] = to_np(hf_model.transformer.wte.weight)
        self.transformer.wpe.weight.data[...] = to_np(hf_model.transformer.wpe.weight)
        
        self.transformer.ln_f.weight.data[...] = to_np(hf_model.transformer.ln_f.weight)
        self.transformer.ln_f.bias.data[...] = to_np(hf_model.transformer.ln_f.bias)
        
        lm_w = to_np(hf_model.lm_head.weight)
        target_w_shape = self.lm_head.weight.data.shape
        if lm_w.T.shape == target_w_shape:
            self.lm_head.weight.data[...] = lm_w.T
        elif lm_w.shape == target_w_shape:
            self.lm_head.weight.data[...] = lm_w
        else:
            raise ValueError(f"LM head weight shape mismatch: got {lm_w.shape}, expected {target_w_shape} or its transpose")
        if self.lm_head.bias is not None:
            if getattr(hf_model.lm_head, "bias", None) is None:
                # HF GPT-2 typically has no lm_head bias; zero ours for parity
                import numpy as _np
                self.lm_head.bias.data[...] = _np.zeros_like(self.lm_head.bias.data)
            else:
                self.lm_head.bias.data[...] = to_np(hf_model.lm_head.bias)
        
        for i in range(len(self.transformer.h)):
            tb = self.transformer.h[i]
            hb = hf_model.transformer.h[i]
            
            tb.ln1.weight.data[...] = to_np(hb.ln_1.weight)
            tb.ln1.bias.data[...] = to_np(hb.ln_1.bias)
            tb.ln2.weight.data[...] = to_np(hb.ln_2.weight)
            tb.ln2.bias.data[...] = to_np(hb.ln_2.bias)
            
            W_fused = to_np(hb.attn.c_attn.weight)
            b_fused = to_np(hb.attn.c_attn.bias)
            
            if W_fused.shape == (3 * self.embed_dim, self.embed_dim):
                W_fused = W_fused.T
            
            Wq, Wk, Wv = np.split(W_fused, 3, axis=1)
            bq, bk, bv = np.split(b_fused, 3, axis=0)
            
            _assign_linear(tb.attn.attn.q_proj, Wq, bq)
            _assign_linear(tb.attn.attn.k_proj, Wk, bk)
            _assign_linear(tb.attn.attn.v_proj, Wv, bv)
            
            W_out = to_np(hb.attn.c_proj.weight)
            b_out = to_np(hb.attn.c_proj.bias)
            _assign_linear(tb.attn.attn.out_proj, W_out, b_out)
            
            W_fc = to_np(hb.mlp.c_fc.weight)
            b_fc = to_np(hb.mlp.c_fc.bias)
            _assign_linear(tb.ffn.linear1, W_fc, b_fc)
            
            W_proj = to_np(hb.mlp.c_proj.weight)
            b_proj = to_np(hb.mlp.c_proj.bias)
            _assign_linear(tb.ffn.linear2, W_proj, b_proj)