import torch
import json
import subprocess
import sys
import numpy as np
from transformers import BertModel

def main():
    # 1. Run the Go script
    print("Running Go weight dump script...")
    try:
        # We need to compile or run. 'go run' works.
        result = subprocess.run(
            ["go", "run", "scripts/dump_weights.go", "-weights", "bert_tiny.bin"],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Go script failed:\n{e.stderr}")
        sys.exit(1)

    go_dumps = json.loads(result.stdout)
    go_map = {d["name"]: d for d in go_dumps}

    # 2. Load PyTorch Model
    print("Loading PyTorch model: prajjwal1/bert-tiny...")
    pt_model = BertModel.from_pretrained("prajjwal1/bert-tiny")
    
    # Map Go names to PyTorch names
    # Go Name -> PyTorch Name
    # Note: Fletcher stores weights as (In, Out) for Linear? Or (Out, In)?
    # PyTorch Linear weights are (Out, In).
    # scripts/download_weights.py DOES transpose Linear weights to (In, Out).
    # So:
    # Go (In, Out)  <-- Transpose --> PyTorch (Out, In)
    # This means Go's "Rows" should match PyTorch "In_Features" (dim 1)
    # And Go's "Cols" should match PyTorch "Out_Features" (dim 0)
    # Wait.
    # device.Tensor Dims() returns (Rows, Cols).
    # If Go matrix is (In, Out), then Rows=In, Cols=Out.
    # PyTorch weight.shape is (Out, In).
    # So Go.Rows == PyTorch.shape[1]
    #    Go.Cols == PyTorch.shape[0]
    
    mapping = {
        "embeddings.word_embeddings": ("embeddings.word_embeddings.weight", False), # Embeddings are (Vocab, Dim) in both usually?
             # PyTorch Embedding: (NumEmbed, Dim). 
             # Fletcher Embedding: (NumEmbed, Dim). No transpose.

        "embeddings.position_embeddings": ("embeddings.position_embeddings.weight", False),
        "embeddings.token_type_embeddings": ("embeddings.token_type_embeddings.weight", False),
        
        "embeddings.LayerNorm.weight": ("embeddings.LayerNorm.weight", False), # 1D
        "embeddings.LayerNorm.bias": ("embeddings.LayerNorm.bias", False),     # 1D

        "encoder.layer.0.attention.self.query.weight": ("encoder.layer.0.attention.self.query.weight", True), # Linear
        "encoder.layer.0.attention.self.query.bias": ("encoder.layer.0.attention.self.query.bias", False),

        "encoder.layer.0.attention.output.dense.weight": ("encoder.layer.0.attention.output.dense.weight", True), # Linear

        "pooler.dense.weight": ("pooler.dense.weight", True), # Linear
    }

    mismatches = 0

    print("\nVerifying Weights...")

    for go_name, (pt_name, is_linear) in mapping.items():
        if go_name not in go_map:
            print(f"MISSING in Go output: {go_name}")
            mismatches += 1
            continue

        go_data = go_map[go_name]
        pt_tensor = pt_model.state_dict()[pt_name].detach().numpy() 

        # 1. Check Dimensions
        if is_linear: # Check Transposed
            # PyTorch: (Out, In)
            # Go: (In, Out)
            expected_rows = pt_tensor.shape[1]
            expected_cols = pt_tensor.shape[0]
            
            # If we are effectively comparing raw values, we need to transpose PyTorch back to match Go
            pt_to_compare = pt_tensor.T
        else:
            # Standard
            if len(pt_tensor.shape) == 1:
                expected_rows = pt_tensor.shape[0]
                expected_cols = 1 # Fletcher treats vectors as (N, 1)? Or flat?
                                  # dump_weights.go: r, c := t.Dims()
                                  # For vectors, CPU backend likely returns (N, 1) or (1, N).
                                  # Let's see what Go outputs.
            else:
                expected_rows = pt_tensor.shape[0]
                expected_cols = pt_tensor.shape[1]
            
            pt_to_compare = pt_tensor

        # Handle Vector Dimension ambiguity in Go Dims()
        # If Go says Cols=1, and PyTorch is 1D (shape=(N,)), we treat Go Rows=N.
        if len(pt_tensor.shape) == 1:
             # PyTorch (N,)
             # Go might be (N, 1)
             if go_data["cols"] == 1:
                 if go_data["rows"] != pt_tensor.shape[0]:
                    print(f"DIM MISMATCH {go_name}: Go ({go_data['rows']}, {go_data['cols']}) vs PyTorch {pt_tensor.shape}")
                    mismatches += 1
             elif go_data["rows"] == 1:
                 if go_data["cols"] != pt_tensor.shape[0]:
                    print(f"DIM MISMATCH {go_name}: Go ({go_data['rows']}, {go_data['cols']}) vs PyTorch {pt_tensor.shape}")
                    mismatches += 1
        else:
            if go_data["rows"] != expected_rows or go_data["cols"] != expected_cols:
                print(f"DIM MISMATCH {go_name}: Go ({go_data['rows']}, {go_data['cols']}) vs PyTorch (Transposed? {is_linear}) {pt_tensor.shape} -> Used ({expected_rows}, {expected_cols})")
                mismatches += 1

        # 2. Check Sum / Content
        # We compare the Flattened array sum
        pt_sum = float(np.sum(pt_to_compare))
        go_sum = go_data["sum"]
        
        # Checking sum similarity
        diff = abs(pt_sum - go_sum)
        rel_diff = diff / (abs(pt_sum) + 1e-9)

        if rel_diff > 1e-4: # Tolerance
             print(f"VALUE MISMATCH {go_name}: Sum Go {go_sum} vs PyTorch {pt_sum} (Diff: {diff})")
             # Print first few for debugging
             flat_pt = pt_to_compare.flatten()
             print(f"  Go First 5: {go_data['first_few']}")
             print(f"  PT First 5: {flat_pt[:5]}")
             mismatches += 1
        else:
             # print(f"OK: {go_name}")
             pass

    if mismatches == 0:
        print("\nSUCCESS: Weights verified!")
    else:
        print(f"\nFAILURE: {mismatches} mismatches.")
        sys.exit(1)

if __name__ == "__main__":
    main()
