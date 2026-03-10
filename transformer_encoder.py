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
    

