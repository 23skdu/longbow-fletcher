
import subprocess
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import sys

cmd = ["./bin/fletcher", "-model", "bert-tiny", "-weights", "bert_tiny.safetensors", "-precision", "fp16", "-input", "fletcher_input.json", "-gpu=true"]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True)

if result.returncode != 0:
    print("Fletcher failed:")
    print(result.stderr.decode())
    sys.exit(1)

print(f"Stdout length: {len(result.stdout)} bytes")

try:
    reader = ipc.open_stream(result.stdout)
    table = reader.read_all()
    print("Arrow Table Read Successfully.")
    print(f"Columns: {table.column_names}")
    
    d = table.column("embedding")
    # d is ChunkedArray of FixedSizeList
    print(f"Embedding Column Type: {d.type}")
    
    # Flatten
    # Since it's a ChunkedArray, we might need to combine chunks
    # But usually one chunk for small output
    if d.num_chunks == 0:
        print("No chunks!")
        sys.exit(1)
        
    chunk0 = d.chunk(0)
    # chunk0 is FixedSizeListArray
    values = chunk0.values # FloatArray
    np_vals = values.to_numpy()
    
    print(f"Values shape: {np_vals.shape}")
    print(f"Values sample: {np_vals[:10]}")
    print(f"Mean: {np.mean(np_vals)}")
    
    if np.mean(np_vals) == 0:
        print("ZEROS DETECTED!")
    else:
        print("Valid Data Detected.")

except Exception as e:
    print(f"Arrow Parsing Failed: {e}")
    # Hex dump head
    print("Hex dump head:")
    print(result.stdout[:100].hex())
