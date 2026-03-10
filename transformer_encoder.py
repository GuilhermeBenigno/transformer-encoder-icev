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

