import numpy as np
import pandas as pd

np.random.seed(42)

vocab_dict = {
    "o":        0,
    "banco":    1,
    "bloqueou": 2,
    "cartao":   3,
    "de":       4,
    "credito":  5,
    "meu":      6,
    "sem":      7,
    "motivo":   8,
}

df_vocab = pd.DataFrame(
    list(vocab_dict.items()),
    columns=["palavra", "id"]
)

print("=== Vocabulário ===")
print(df_vocab.to_string(index=False))
print()

frase = ["o", "banco", "bloqueou", "meu", "cartao", "de", "credito"]
input_ids = [vocab_dict[palavra] for palavra in frase]

print(f"Frase de entrada : {frase}")
print(f"IDs da frase     : {input_ids}")
print()
vocab_size     = len(vocab_dict)
embedding_table = np.random.randn(vocab_size, D_MODEL)

print(f"Frase de entrada : {frase}")
print(f"IDs da frase     : {input_ids}")

X = embedding_table[input_ids]
X = X[np.newaxis, :, :]

print(f"Shape do tensor de entrada X : {X.shape}  →  (Batch, Tokens, d_model)")
print()

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x     = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(X, WQ, WK, WV):
    d_k = WK.shape[1]
    
    Q = X @ WQ
    K = X @ WK   
    V = X @ WV
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    
    attn_weights = softmax(scores)
    
    output = attn_weights @ V
    return output

def layer_norm(x, epsilon=EPSILON):
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x,  axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon)

def feed_forward_network(x, W1, b1, W2, b2):
    hidden = np.maximum(0, x @ W1 + b1)
    output = hidden @ W2 + b2
    return output

layer_weights = []
for i in range(N_LAYERS):
    d_k = D_MODEL
     d_v = D_MODEL
     WQ = np.random.randn(D_MODEL, d_k) * np.sqrt(2.0 / D_MODEL)
     WK = np.random.randn(D_MODEL, d_k) * np.sqrt(2.0 / D_MODEL)
     WV = np.random.randn(D_MODEL, d_v) * np.sqrt(2.0 / D_MODEL)

     W1 = np.random.randn(D_MODEL, D_FF)   * np.sqrt(2.0 / D_MODEL)
     b1 = np.zeros(D_FF)
     W2 = np.random.randn(D_FF,   D_MODEL) * np.sqrt(2.0 / D_FF)
     b2 = np.zeros(D_MODEL)

    layer_weights.append((WQ, WK, WV, W1, b1, W2, b2))


    

