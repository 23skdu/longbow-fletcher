import torch
from transformers import BertModel, BertTokenizer
import json
import numpy as np

def main():
    model_name = "prajjwal1/bert-tiny"
    print(f"Loading {model_name}...")
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    
    texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Fletcher is a high performance embedding engine",
        "Machine learning is fascinating"
    ]
    
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            # Fletcher uses [CLS] token + Pooler (Dense + Tanh)
            # output.pooler_output provides exactly this
            # UNLESS model has no pooler.
            if outputs.pooler_output is not None:
                emb = outputs.pooler_output.numpy()[0]
                # Check range (should be -1 to 1)
                # print(f"First 5: {emb[:5]}")
            else:
                # Fallback to CLS token if no pooler (rare for BERT)
                emb = outputs.last_hidden_state[:, 0, :].numpy()[0]
            
            embeddings.append(emb.tolist())
            
    output = {
        "model": model_name,
        "texts": texts,
        "embeddings": embeddings
    }
    
    with open("hf_reference.json", "w") as f:
        json.dump(output, f, indent=2)
        
    print("Ref embeddings saved to hf_reference.json using raw BertModel")

if __name__ == "__main__":
    main()
