# Supported Models and Dimensions

| Model | Hidden Size / Dim | Notes |
|-------|-------------------|-------|
| bert-tiny | 128 | Fast baseline |
| all-MiniLM-L6-v2 | 384 | General-purpose |
| nomic-embed-text | 768 | RoPE, SwiGLU, task prefixes |
| roberta-base | 768 | RoBERTa |
| xlm-roberta-base | 768 | Multilingual |
| bge-m3 | 1024 | Multilingual |
| e5-mistral-7b-instruct | 4096 | Instruction-tuned |

Use matching dimensions when creating Longbow datasets.