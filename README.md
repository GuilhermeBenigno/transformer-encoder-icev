Descrição

Implementação do **Forward Pass** de um Encoder Transformer completo com **N = 6 camadas**, conforme o paper original *"Attention Is All You Need"* (Vaswani et al., 2017).

Ferramentas utilizadas: `Python 3.x`, `numpy`, `pandas`.  
**Não foi utilizado** PyTorch, TensorFlow, Keras ou qualquer biblioteca de atenção pronta.

---

---

## Estrutura do Projeto

```
.
├── transformer_encoder.py   # Implementação completa do Encoder
└── README.md
```

---

## Como Rodar

### Pré-requisitos

```bash
pip install numpy pandas
```

### Execução

```bash
python transformer_encoder.py
```

A saída esperada mostra:
- O vocabulário em um DataFrame pandas
- A frase convertida para IDs
- O shape do tensor de entrada `(1, 7, 64)`
- O shape de saída após cada uma das 6 camadas (deve ser idêntico)
- A validação de sanidade confirmando `(Batch, Tokens, d_model)` preservado

---

## Arquitetura Implementada

### Hiperparâmetros

| Parâmetro | Paper Original | Este Laboratório |
|-----------|---------------|-----------------|
| `d_model` | 512 | 64 |
| `d_ff`    | 2048 | 256 |
| `N`       | 6 | 6 |

### Componentes por Camada

```
X  ──► Self-Attention ──► Add & LayerNorm ──► FFN ──► Add & LayerNorm ──► X_out
       (Scaled Dot-         (Residual)         (ReLU)   (Residual)
        Product)
```

1. **Scaled Dot-Product Attention**  
   `Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`

2. **Add & LayerNorm** (pós-atenção)  
   `X_norm1 = LayerNorm(X + X_att)`

3. **Feed-Forward Network**  
   `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`

4. **Add & LayerNorm** (pós-FFN)  
   `X_out = LayerNorm(X_norm1 + X_ffn)`

---

## Nota de Crédito

Ferramenta de IA (Claude – Anthropic) foi consultada como apoio para revisão de sintaxe NumPy e broadcasting. O código foi entendido, adaptado e validado pelo aluno.
