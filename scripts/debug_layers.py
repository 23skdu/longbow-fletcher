import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import sys
import os

MODEL_NAME = "prajjwal1/bert-tiny"
INPUT_FILE = "fletcher_input.json"

def load_tensor(path, shape):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data.reshape(shape)

def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Load input
    # We used N=4 lowercase input
    sentences = [
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        "fletcher is a high performance embedding engine",
        "machine learning is fascinating"
    ]
    
    print("Running PyTorch Forward Pass...")
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    # Hook intermediate layers
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output.detach().numpy()
        return hook

    # Register hooks
    # Embeddings
    model.embeddings.register_forward_hook(get_activation("embeddings"))
    # Layers
    for i, layer in enumerate(model.encoder.layer):
        layer.register_forward_hook(get_activation(f"layer_{i}"))
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    pooler_output = outputs.pooler_output.numpy()
    
    print("\n--- Comparison ---")
    
    # 1. Embeddings
    # Fletcher dump is flat 
    # Shape: (Batch, Seq, Hidden) -> (4, SeqLen, 128)
    # But Fletcher handles padding? Or just raw sequences?
    # Fletcher output is usually padded if batched?
    # No, Fletcher internal tensors are concatenated? 
    # Wait, in ForwardBatch, embeddings is (Batch*Seq, Hidden) or (Batch, Seq, Hidden)?
    # Metal backend usually flattens?
    # Actually, let's assume we can infer shape from file size / 128
    
    attention_mask = inputs['attention_mask'].numpy() # (Batch, Seq)
    
    def compare(name, pytorch_data, fletcher_file):
        if not os.path.exists(fletcher_file):
            print(f"[{name}] File {fletcher_file} not found!")
            return

        ft_data = np.fromfile(fletcher_file, dtype=np.float32)
        
        # If data is 2D (Batch, Hidden), skip masking
        if len(pytorch_data.shape) == 2:
            pt_flat = pytorch_data.flatten()
            ft_flat = ft_data.flatten()
        else:
            # Filter PyTorch data using mask
            # ...
            
            mask_expanded = attention_mask[:, :, np.newaxis]
            pt_valid = pytorch_data[attention_mask == 1]
            pt_flat = pt_valid.flatten()
            ft_flat = ft_data.flatten()
        
        if len(pt_flat) != len(ft_flat):
            print(f"[{name}] Size Mismatch! PT={len(pt_flat)} (Valid Tokens), FT={len(ft_flat)}")
            return
            
        # Cosine Sim
        sim = np.dot(pt_flat, ft_flat) / (np.linalg.norm(pt_flat) * np.linalg.norm(ft_flat))
        print(f"[{name}] Cosine Similarity: {sim:.5f}")
        
    compare("Embeddings", activations["embeddings"], "debug_embeddings.bin")
    compare("Layer 0", activations["layer_0"], "debug_layer_0.bin")
    compare("Layer 1", activations["layer_1"], "debug_layer_1.bin")
    
    # Compare IDs
    if os.path.exists("debug_ids.bin"):
        ft_ids = np.fromfile("debug_ids.bin", dtype=np.int32)
        print(f"\n[IDs] Fletcher IDs (First 10): {ft_ids[:10]}")
        # PyTorch IDs (masked)
        pt_ids = inputs['input_ids'].numpy()[attention_mask == 1].flatten()
        print(f"[IDs] PyTorch IDs  (First 10): {pt_ids[:10]}")
        
        if len(ft_ids) == len(pt_ids):
             mismatches = np.sum(ft_ids != pt_ids)
             print(f"[IDs] Mismatches: {mismatches} / {len(ft_ids)}")
        else:
             print(f"[IDs] Length Mismatch! PT={len(pt_ids)}, FT={len(ft_ids)}")
    else:
        print("\n[IDs] debug_ids.bin not found!")
    
    
    # Pooler is (Batch, Hidden). No masking needed (1 vector per seq)
    # But Fletcher assumes flat.
    # We should just flatten Pooler.
    compare("Pooler", pooler_output, "debug_pooler.bin")

if __name__ == "__main__":
    main()
