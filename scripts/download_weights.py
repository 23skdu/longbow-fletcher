import torch
from transformers import BertModel, BertTokenizer
import struct
import os

def main():
    model_name = "prajjwal1/bert-tiny"
    print(f"Downloading {model_name}...")
    
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Save Vocab
    print("Saving vocab.txt...")
    tokenizer.save_pretrained(".")
    
    # Extract weights and save to binary
    output_file = "bert_tiny.bin"
    print(f"Saving weights to {output_file}...")
    
    with open(output_file, "wb") as f:
        # Helper to write tensor
        def write_tensor(tensor):
            # Flatten and write as float32 little endian
            data = tensor.detach().cpu().numpy().flatten().astype("float32")
            f.write(data.tobytes())
            
        # 1. Embeddings
        # word_embeddings
        write_tensor(model.embeddings.word_embeddings.weight)
        # position_embeddings
        write_tensor(model.embeddings.position_embeddings.weight)
        # token_type_embeddings
        write_tensor(model.embeddings.token_type_embeddings.weight)
        # LayerNorm
        write_tensor(model.embeddings.LayerNorm.weight) # Gamma
        write_tensor(model.embeddings.LayerNorm.bias)   # Beta
        
        # 2. Encoder
        for layer in model.encoder.layer:
            # Attention Self
            # Transpose Linear weights: HF (Out, In) -> Go (In, Out)
            write_tensor(layer.attention.self.query.weight.t())
            write_tensor(layer.attention.self.query.bias)
            write_tensor(layer.attention.self.key.weight.t())
            write_tensor(layer.attention.self.key.bias)
            write_tensor(layer.attention.self.value.weight.t())
            write_tensor(layer.attention.self.value.bias)
            
            # Attention Output
            write_tensor(layer.attention.output.dense.weight.t())
            write_tensor(layer.attention.output.dense.bias)
            write_tensor(layer.attention.output.LayerNorm.weight)
            write_tensor(layer.attention.output.LayerNorm.bias)
            
            # Intermediate
            write_tensor(layer.intermediate.dense.weight.t())
            write_tensor(layer.intermediate.dense.bias)
            
            # Output
            write_tensor(layer.output.dense.weight.t())
            write_tensor(layer.output.dense.bias)
            write_tensor(layer.output.LayerNorm.weight)
            write_tensor(layer.output.LayerNorm.bias)
            
        # 3. Pooler
        if model.pooler is not None:
            write_tensor(model.pooler.dense.weight.t())
            write_tensor(model.pooler.dense.bias)
        else:
            print("Warning: Model has no pooler!")
            
    print("Done.")

if __name__ == "__main__":
    main()
