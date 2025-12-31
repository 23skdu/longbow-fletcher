import subprocess
import json
import numpy as np
import sys
import os
import pyarrow as pa
import pyarrow.ipc as ipc
from sklearn.metrics.pairwise import cosine_similarity
import io

def main():
    # Load reference
    try:
        with open("hf_reference.json", "r") as f:
            ref_data = json.load(f)
    except FileNotFoundError:
        print("hf_reference.json not found")
        sys.exit(1)
        
    texts = ref_data["texts"]
    ref_embeds = np.array(ref_data["embeddings"])
    
    # Save input for fletcher
    with open("fletcher_input.json", "w") as f:
        json.dump(texts, f)
        
    print(f"Running Fletcher with {len(texts)} inputs...")
    
    # Run Fletcher: go run ./cmd/fletcher -input fletcher_input.json -gpu
    # Capture stdout which should be Arrow IPC stream
    # Add Accelerate framework for macOS BLAS
    env = os.environ.copy()
    env["CGO_LDFLAGS"] = "-framework Accelerate"
    cmd = ["go", "run", "./cmd/fletcher", "-input", "fletcher_input.json"]
    
    # Check if we should use GPU (optional, based on env or flag, defaulted to CPU for stability unless user requested GPU explicitly)
    # The user request mentioned "metal-kernel-optimization", implying testing on Metal.
    # But for verification of numeric correctness vs HF, CPU backend is safer baseline first?
    # Let's try CPU first as default.
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Fletcher failed: {e.stderr.decode()}")
        sys.exit(1)
        
    # Read Arrow Stream from stdout
    try:
        reader = ipc.open_stream(io.BytesIO(result.stdout))
        table = reader.read_all()
    except Exception as e:
        print(f"Failed to read Arrow stream: {e}")
        print("Stdout preview:", result.stdout[:200])
        sys.exit(1)
        
    # Extract embeddings
    # Schema: text (utf8), embedding (fixed_size_list)
    # Convert to numpy
    
    # Arrow table to pandas or directly
    embed_col = table.column("embedding")
    
    # Flatten and reshape
    # embed_col is a ChunkedArray of FixedSizeList
    # We need to extract values.
    
    # PyArrow 14+ specific handling might be needed, but generally:
    # Convert to numpy stack
    fletcher_embeds = np.stack(embed_col.to_numpy())
    
    print(f"Got embeddings shape: {fletcher_embeds.shape}")
    
    # Compare
    sims = cosine_similarity(ref_embeds, fletcher_embeds)
    diag_sims = np.diag(sims)
    
    print(f"Cosine Similarities (diagonal): {diag_sims}")
    
    print("\nReference Vector 0 (first 5):", ref_embeds[0][:5])
    print("Fletcher Vector 0 (first 5):", fletcher_embeds[0][:5])
    
    avg_sim = np.mean(diag_sims)
    print(f"Average Similarity: {avg_sim:.4f}")
    
    if avg_sim < 0.99:
        print("Verification FAILED: Similarity too low")
        sys.exit(1)
        
    print("Verification PASSED")

if __name__ == "__main__":
    main()
