import torch
import json
import subprocess
import sys
import numpy as np
from transformers import BertModel, BertTokenizer

def main():
    text = "hello world"
    model_name = "prajjwal1/bert-tiny"
    
    # 1. Run Go Script
    print(f"Running Go embedding debug for '{text}'...")
    try:
        result = subprocess.run(
            ["go", "run", "scripts/debug_embeddings.go", "-text", text, "-weights", "bert_tiny.bin"],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Go script failed:\n{e.stderr}")
        sys.exit(1)

    go_dumps = {d["name"]: d for d in json.loads(result.stdout)}

    # 2. PyTorch Reference
    print(f"Running PyTorch reference ({model_name})...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"] 
    # Note: HuggingFace default might add [CLS] [SEP]. verify.
    # print(f"PT Input IDs: {input_ids}")

    # A. Word Embeddings
    pt_word_embeds = model.embeddings.word_embeddings(input_ids)
    
    # B. Position Embeddings
    # Create position IDs 0..seq_len
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    pt_pos_embeds = model.embeddings.position_embeddings(position_ids)

    # C. Token Type Embeddings
    token_type_ids = torch.zeros((1, seq_len), dtype=torch.long)
    pt_type_embeds = model.embeddings.token_type_embeddings(token_type_ids)

    # D. Sum
    pt_sum_embeds = pt_word_embeds + pt_pos_embeds + pt_type_embeds

    # E. LayerNorm
    pt_ln_embeds = model.embeddings.LayerNorm(pt_sum_embeds)

    # Comparison Helper
    def compare(name, pt_tensor, go_dump):
        # PT is (1, Seq, Hidden). Go is (Seq, Hidden) flattened or 2D.
        pt_flat = pt_tensor.detach().numpy().flatten()
        go_data = np.array(go_dump["values"])
        
        if len(pt_flat) != len(go_data):
            print(f"MISMATCH SHAPE {name}: PT {pt_flat.shape} vs Go {go_data.shape}")
            return False

        # Cosine Similarity
        dot = np.dot(pt_flat, go_data)
        norm_pt = np.linalg.norm(pt_flat)
        norm_go = np.linalg.norm(go_data)
        sim = dot / (norm_pt * norm_go)
        
        # Max Diff
        diff = np.abs(pt_flat - go_data)
        max_diff = np.max(diff)

        print(f"CHECK {name}: Sim={sim:.6f}, MaxDiff={max_diff:.6f}")
        
        if sim < 0.9999: # Strict check for embeddings
            return False
        return True

    failed = False
    if not compare("word_embeddings", pt_word_embeds, go_dumps["word_embeddings"]): failed = True
    if not compare("position_embeddings", pt_pos_embeds, go_dumps["position_embeddings"]): failed = True
    if not compare("token_type_embeddings", pt_type_embeds, go_dumps["token_type_embeddings"]): failed = True
    if not compare("sum_embeddings", pt_sum_embeds, go_dumps["sum_embeddings"]): failed = True
    if not compare("layernorm_embeddings", pt_ln_embeds, go_dumps["layernorm_embeddings"]): failed = True

    if failed:
        print("\nFAILURE: Embedding layer mismatch.")
        sys.exit(1)
    else:
        print("\nSUCCESS: Embeddings Verified!")

if __name__ == "__main__":
    main()
