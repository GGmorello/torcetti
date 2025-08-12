import numpy as np
from torcetti.core.tensor import Tensor

def softmax(input, dim=-1):
    axis = dim
    max_vals = np.max(input.data, axis=axis, keepdims=True)
    shifted = input.data - max_vals
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals, axis=axis, keepdims=True)
    softmax_output = exp_vals / sum_exp
    
    out = Tensor(softmax_output, input.requires_grad, _children=(input,), _op='softmax')
    
    def _backward():
        s_dot_grad = softmax_output * out.grad.data
        sum_s_dot_grad = np.sum(s_dot_grad, axis=axis, keepdims=True)
        grad_contribution = softmax_output * (out.grad.data - sum_s_dot_grad)
        input.grad += grad_contribution.astype(input.data.dtype)
    
    out._backward = _backward
    return out

def log_softmax(input, dim=-1):
    axis = dim
    max_vals = np.max(input.data, axis=axis, keepdims=True)
    shifted = input.data - max_vals
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    log_softmax_output = shifted - log_sum_exp
    
    out = Tensor(log_softmax_output, input.requires_grad, _children=(input,), _op='log_softmax')
    
    def _backward():
        softmax_vals = np.exp(log_softmax_output)
        sum_grad = np.sum(out.grad.data, axis=axis, keepdims=True)
        input.grad += out.grad.data - softmax_vals * sum_grad
    
    out._backward = _backward
    return out

def relu(input):
    output_data = np.maximum(input.data, 0).astype(input.data.dtype)
    out = Tensor(output_data, input.requires_grad, _children=(input,), _op='relu')
    
    def _backward():
        mask = (input.data > 0).astype(input.data.dtype)
        input.grad += mask * out.grad.data
    
    out._backward = _backward
    return out

def gelu(input, approximate=False):
    import math
    if approximate:
        x = input.data
        c = np.sqrt(2 / np.pi)
        c1 = 0.044715
        c2 = c * (1 + c1 * 3 * x**2)
        tanh_term = np.tanh(c * (x + c1 * x**3))
        output_data = 0.5 * x * (1 + tanh_term)
    else:
        x = input.data
        cdf = 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))
        output_data = x * cdf
    out = Tensor(output_data, input.requires_grad, _children=(input,), _op='gelu')
    
    def _backward():
        if approximate:
            input.grad += (0.5 + 0.5 * np.tanh(np.sqrt(2 / np.pi) * (input.data + 0.044715 * input.data**3))) * out.grad.data
        else:
            x = input.data
            cdf = 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))
            pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
            grad_input = cdf + x * pdf
            input.grad += grad_input * out.grad.data
    
    out._backward = _backward
    return out

def sigmoid(input):
    sigmoid_output = 1 / (1 + np.exp(-input.data))
    out = Tensor(sigmoid_output, input.requires_grad, _children=(input,), _op='sigmoid')
    
    def _backward():
        input.grad += (sigmoid_output * (1 - sigmoid_output)) * out.grad.data
    
    out._backward = _backward
    return out

def tanh(input):

    tanh_output = np.tanh(input.data)
    out = Tensor(tanh_output, input.requires_grad, _children=(input,), _op='tanh')
    
    def _backward():
        input.grad += (1 - tanh_output**2) * out.grad.data
    
    out._backward = _backward
    return out 

def dropout(input, p=0.5, training=True):
    if p == 0 or not training:
        out = Tensor(input.data, input.requires_grad, _children=(input,), _op='dropout')
        def _backward():
            if input.requires_grad:
                input.grad += out.grad.data
        out._backward = _backward
        return out
    else:
        prob_keep = 1 - p           
        rand = np.random.rand(*input.data.shape)
        mask = (rand < prob_keep).astype(input.data.dtype)
        
        output_data = input.data * mask / (1 - p)
        out = Tensor(output_data, input.requires_grad, _children=(input,), _op='dropout')
        
        def _backward():
            input.grad += (mask / (1 - p)) * out.grad.data
        
        out._backward = _backward
        return out

def batch_norm(input, running_mean, running_var, eps=1e-5, momentum=0.1, training=True):
    if training:
        batch_mean = input.mean(dim=0)
        batch_var = input.var(dim=0, ddof=0)
        x_centered = input - batch_mean
        inv_std = 1.0 / np.sqrt(batch_var.data + eps)
        x_norm_data = x_centered.data * inv_std
        running_mean.data = (1 - momentum) * running_mean.data + momentum * batch_mean.data
        running_var.data = (1 - momentum) * running_var.data + momentum * batch_var.data

        out = Tensor(x_norm_data, input.requires_grad, _children=(input,), _op='batch_norm')
        
        def _backward():
            N = input.data.shape[0]
            dx_normalized = out.grad.data
            x_centered_data = x_centered.data
            std_inv = 1.0 / np.sqrt(batch_var.data + eps)
            dx_var = -0.5 * std_inv**3 * np.sum(x_centered_data * dx_normalized, axis=0)
            dx_mean = -np.sum(dx_normalized, axis=0) * std_inv + dx_var * (-2.0 / N) * np.sum(x_centered_data, axis=0)
            dx = dx_normalized * std_inv + dx_var * (2.0 / N) * x_centered_data + dx_mean / N
            input.grad += dx
        
        out._backward = _backward
        return out
    else:
        inv_std = 1.0 / np.sqrt(running_var.data + eps)
        x_norm_data = (input - running_mean).data * inv_std

        out = Tensor(x_norm_data, input.requires_grad, _children=(input,), _op='batch_norm')
        
        def _backward():
            std_inv = 1.0 / np.sqrt(running_var.data + eps)
            input.grad += out.grad.data * std_inv
        
        out._backward = _backward
        return out

def conv2d(input, weight, bias=None, stride=1, padding=0):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    N, C_in, H_in, W_in = input.data.shape
    C_out, C_in_w, K_H, K_W = weight.data.shape
    
    assert C_in == C_in_w, f"Input channels mismatch: {C_in} vs {C_in_w}"
    
    dtype = input.data.dtype
    
    if padding[0] > 0 or padding[1] > 0:
        padded_input = np.pad(input.data, 
                            ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 
                            mode='constant', constant_values=0.0).astype(dtype)
    else:
        padded_input = input.data
    
    H_padded, W_padded = padded_input.shape[2], padded_input.shape[3]
    H_out = (H_padded - K_H) // stride[0] + 1
    W_out = (W_padded - K_W) // stride[1] + 1
    
    output_data = np.zeros((N, C_out, H_out, W_out), dtype=dtype)
    
    weight_data = weight.data.astype(dtype)
    
    for n in range(N):
        for c_out in range(C_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride[0]
                    h_end = h_start + K_H
                    w_start = w_out * stride[1]
                    w_end = w_start + K_W
                    
                    patch = padded_input[n, :, h_start:h_end, w_start:w_end]
                    val = np.sum(patch.astype(np.float64) * weight_data[c_out].astype(np.float64))
                    # Assign; array dtype will handle casting
                    output_data[n, c_out, h_out, w_out] = val
    
    if bias is not None:
        bias_data = bias.data.astype(dtype)
        output_data += bias_data.reshape(1, -1, 1, 1)
    
    requires_grad = input.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    children = (input, weight) if bias is None else (input, weight, bias)
    out = Tensor(output_data, requires_grad, _children=children, _op='conv2d')
    
    def _backward():
        if input.requires_grad:
            input_grad = np.zeros_like(input.data)
            
            if padding[0] > 0 or padding[1] > 0:
                padded_input_grad = np.zeros_like(padded_input, dtype=dtype)
            else:
                padded_input_grad = input_grad
            
            for n in range(N):
                for c_out in range(C_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            h_start = h_out * stride[0]
                            h_end = h_start + K_H
                            w_start = w_out * stride[1]
                            w_end = w_start + K_W
                            
                            grad_val = out.grad.data[n, c_out, h_out, w_out].astype(dtype)
                            padded_input_grad[n, :, h_start:h_end, w_start:w_end] += weight_data[c_out] * grad_val
            
            if padding[0] > 0 or padding[1] > 0:
                input.grad += padded_input_grad[:, :, padding[0]:H_padded-padding[0], padding[1]:W_padded-padding[1]]
            else:
                input.grad += padded_input_grad
        
        if weight.requires_grad:
            weight_grad = np.zeros_like(weight.data, dtype=dtype)
            
            for n in range(N):
                for c_out in range(C_out):
                    for h_out in range(H_out):
                        for w_out in range(W_out):
                            h_start = h_out * stride[0]
                            h_end = h_start + K_H
                            w_start = w_out * stride[1]
                            w_end = w_start + K_W
                            
                            patch = padded_input[n, :, h_start:h_end, w_start:w_end]
                            grad_val = out.grad.data[n, c_out, h_out, w_out].astype(dtype)
                            weight_grad[c_out] += patch * grad_val
            
            weight.grad += weight_grad
        
        if bias is not None and bias.requires_grad:
            bias_grad = np.sum(out.grad.data, axis=(0, 2, 3)).astype(dtype)
            bias.grad += bias_grad
    
    out._backward = _backward
    return out

def _pair(val):
    return (val, val) if isinstance(val, int) else tuple(val)


def avg_pool2d(input, kernel_size, stride=None, padding=0):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride if stride is not None else kernel_size)
    padding = _pair(padding)

    N, C, H, W = input.data.shape
    kH, kW = kernel_size
    sH, sW = stride

    if padding != (0, 0):
        padded = np.pad(input.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant', constant_values=0.0)
    else:
        padded = input.data

    H_pad, W_pad = padded.shape[2], padded.shape[3]
    H_out = (H_pad - kH) // sH + 1
    W_out = (W_pad - kW) // sW + 1

    out_data = np.zeros((N, C, H_out, W_out), dtype=input.data.dtype)

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    hs = h * sH
                    ws = w * sW
                    patch = padded[n, c, hs:hs + kH, ws:ws + kW]
                    out_data[n, c, h, w] = patch.mean()

    out = Tensor(out_data, input.requires_grad, _children=(input,), _op='avg_pool2d')

    def _backward():
        if not input.requires_grad:
            return
        grad = out.grad.data / (kH * kW)
        padded_grad = np.zeros_like(padded)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        hs = h * sH
                        ws = w * sW
                        padded_grad[n, c, hs:hs + kH, ws:ws + kW] += grad[n, c, h, w]

        if padding != (0, 0):
            input.grad += padded_grad[:, :, padding[0]:padding[0] + H, padding[1]:padding[1] + W]
        else:
            input.grad += padded_grad

    out._backward = _backward
    return out


def max_pool2d(input, kernel_size, stride=None, padding=0):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride if stride is not None else kernel_size)
    padding = _pair(padding)

    N, C, H, W = input.data.shape
    kH, kW = kernel_size
    sH, sW = stride

    if padding != (0, 0):
        pad_val = -np.inf if np.issubdtype(input.data.dtype, np.floating) else np.iinfo(input.data.dtype).min
        padded = np.pad(input.data, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant', constant_values=pad_val)
    else:
        padded = input.data

    H_pad, W_pad = padded.shape[2], padded.shape[3]
    H_out = (H_pad - kH) // sH + 1
    W_out = (W_pad - kW) // sW + 1

    out_data = np.zeros((N, C, H_out, W_out), dtype=input.data.dtype)
    mask = np.zeros_like(padded, dtype=bool)  # store max positions

    for n in range(N):
        for c in range(C):
            for h in range(H_out):
                for w in range(W_out):
                    hs = h * sH
                    ws = w * sW
                    patch = padded[n, c, hs:hs + kH, ws:ws + kW]
                    max_val = patch.max()
                    out_data[n, c, h, w] = max_val
                    max_mask = (patch == max_val)
                    mask[n, c, hs:hs + kH, ws:ws + kW] |= max_mask

    out = Tensor(out_data, input.requires_grad, _children=(input,), _op='max_pool2d')

    def _backward():
        if not input.requires_grad:
            return
        padded_grad = np.zeros_like(padded)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        hs = h * sH
                        ws = w * sW
                        grad_val = out.grad.data[n, c, h, w]
                        window_mask = mask[n, c, hs:hs + kH, ws:ws + kW]
                        padded_grad[n, c, hs:hs + kH, ws:ws + kW] += grad_val * window_mask

        if padding != (0, 0):
            input.grad += padded_grad[:, :, padding[0]:padding[0] + H, padding[1]:padding[1] + W]
        else:
            input.grad += padded_grad

    out._backward = _backward
    return out

def embedding(x, weight):
    indices = x.data.astype(int)
    
    if np.any(indices < 0):
        raise IndexError("Embedding indices must be non-negative")
    
    if np.any(indices >= weight.data.shape[0]):
        raise IndexError("Embedding index out of bounds")
    
    return weight[indices]

def layer_norm(x: Tensor, normalized_shape, weight=None, bias=None, eps: float = 1e-5):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    normalized_shape = tuple(normalized_shape)

    axes = tuple(range(-len(normalized_shape), 0)) if normalized_shape else ()

    mean = x.mean(dim=axes, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=axes, keepdim=True)

    normalized = (x - mean) / (var + eps).sqrt()

    if weight is not None:
        normalized = normalized * weight 
    if bias is not None:
        normalized = normalized + bias

    return normalized



def scaled_dot_product_attention(q: Tensor,
                                 k: Tensor,
                                 v: Tensor,
                                 *,
                                 num_heads: int,
                                 dropout_p: float = 0.0,
                                 attn_mask=None,
                                 key_padding_mask=None,
                                 training: bool = True):
    """Computes scaled dot-product attention using vectorized operations."""
    _, _, d_k = q.shape
    scale = np.sqrt(float(d_k))

    # (B*H, L_q, D) @ (B*H, D, L_k) -> (B*H, L_q, L_k)
    attn_scores = (q @ k.permute(0, 2, 1)) / scale

    if attn_mask is not None:
        attn_scores += attn_mask

    if key_padding_mask is not None:
        B = key_padding_mask.shape[0]
        H = num_heads
        L_q = q.shape[1]
        L_k = k.shape[1]
        
        mask = key_padding_mask.reshape(B, 1, 1, L_k)
        mask = mask.expand(B, H, L_q, L_k)
        mask = mask.reshape(B * H, L_q, L_k)
        
        attn_scores = Tensor.where(~mask, attn_scores, -1e9)

    attn_weights = softmax(attn_scores, dim=-1)

    if dropout_p > 0.0 and training:
        attn_weights = dropout(attn_weights, p=dropout_p, training=training)

    output = attn_weights @ v

    return output, attn_weights
def linear(input, weight, bias=None):

    original_shape = input.shape
    in_features, out_features = weight.shape

    x_2d = input.reshape(-1, in_features)

    out_2d = x_2d @ weight
    if bias is not None:
        out_2d = out_2d + bias

    out = out_2d.reshape(*original_shape[:-1], out_features)
    return out
def multi_head_attention(query: Tensor,
                         key:   Tensor,
                         value: Tensor,
                         *,
                         num_heads: int,
                         num_kv_heads: int = None,
                         head_dim: int,
                         dropout_p: float = 0.0,
                         out_proj: callable,
                         attn_mask=None,
                         key_padding_mask=None,
                         batch_first: bool = False,
                         training: bool = True):

    if num_kv_heads is None:
        num_kv_heads = num_heads
    
    heads_per_kv_head = num_heads // num_kv_heads
    
    def _reshape_query_for_attention(q: Tensor) -> Tensor:
        if batch_first:
            N, L, _ = q.shape
            q = q.reshape(N, L, num_heads, head_dim)
            q = q.permute(0, 2, 1, 3)  # [N, H, L, D]
        else:
            L, N, _ = q.shape
            q = q.reshape(L, N, num_heads, head_dim)
            q = q.permute(1, 2, 0, 3)  # [N, H, L, D]
        return q

    def _reshape_kv_for_attention(x: Tensor) -> Tensor:
        if batch_first:
            N, L, _ = x.shape
            x = x.reshape(N, L, num_kv_heads, head_dim)
            x = x.permute(0, 2, 1, 3)  # [N, KV_H, L, D]
        else:
            L, N, _ = x.shape
            x = x.reshape(L, N, num_kv_heads, head_dim)
            x = x.permute(1, 2, 0, 3)  # [N, KV_H, L, D]
        return x

    def _reshape_back(x: Tensor) -> Tensor:
        N, H, L, D = x.shape
        if batch_first:
            x = x.permute(0, 2, 1, 3)  # [N, L, H, D]
            x = x.reshape(N, L, H * D)
        else:
            x = x.permute(2, 0, 1, 3)  # [L, N, H, D]
            x = x.reshape(L, N, H * D)
        return x

    # Reshape tensors
    q = _reshape_query_for_attention(query)  # [N, H, L_q, D]
    k = _reshape_kv_for_attention(key)       # [N, KV_H, L_k, D]
    v = _reshape_kv_for_attention(value)     # [N, KV_H, L_k, D]

    B, H, L_q, D = q.shape
    _, KV_H, L_k, _ = k.shape
    
    B, KV_H, L_k, D = k.shape
    
    # [B, KV_H, L_k, D] -> [B, KV_H, 1, L_k, D] -> [B, KV_H, heads_per_kv_head, L_k, D] -> [B, H, L_k, D]
    k_expanded = k.unsqueeze(2).repeat(1, 1, heads_per_kv_head, 1, 1).reshape(B, H, L_k, D)
    v_expanded = v.unsqueeze(2).repeat(1, 1, heads_per_kv_head, 1, 1).reshape(B, H, L_k, D)
    

    q_flat = q.reshape(B * H, L_q, D)
    k_flat = k_expanded.reshape(B * H, L_k, D)
    v_flat = v_expanded.reshape(B * H, L_k, D)

    attn_out_flat, attn_weights_flat = scaled_dot_product_attention(
        q_flat,
        k_flat,
        v_flat,
        num_heads=num_heads,
        dropout_p=dropout_p,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        training=training,
    )

    attn_out = attn_out_flat.reshape(B, H, L_q, D)
    attn_weights = attn_weights_flat.reshape(B, H, L_q, L_k)

    out = _reshape_back(attn_out)
    out = out_proj(out)

    return out, attn_weights
