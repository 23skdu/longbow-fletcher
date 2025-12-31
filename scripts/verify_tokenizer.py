import json
import subprocess
import sys
from transformers import BertTokenizer

def main():
    # 1. Run the Go script
    print("Running Go tokenizer script...")
    try:
        # Check if debug_tokenizer_json.go exists? It should.
        # We run it using 'go run'
        result = subprocess.run(
            ["go", "run", "scripts/debug_tokenizer_json.go"],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Go script failed:\n{e.stderr}")
        sys.exit(1)

    go_output = json.loads(result.stdout)
    
    # 2. Load Reference Tokenizer
    model_name = "prajjwal1/bert-tiny"
    print(f"Loading reference tokenizer: {model_name}...")
    hf_tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # 3. Compare
    mismatches = 0
    print("\nVerifying...")
    for case in go_output["cases"]:
        text = case["text"]
        go_ids = case["ids"]
        go_tokens = case["tokens"]
        
        # Fletcher's Tokenize() returns just the tokens for the text.
        # It does NOT add [CLS] and [SEP] by itself (Embedder does that).
        # So we verify against hf_tokenizer.tokenize() / convert_tokens_to_ids()
        
        hf_tokens = hf_tokenizer.tokenize(text)
        hf_ids = hf_tokenizer.convert_tokens_to_ids(hf_tokens)
        
        if go_ids != hf_ids:
            print(f"MISMATCH for '{text}'")
            print(f"  Go IDs: {go_ids}")
            print(f"  HF IDs: {hf_ids}")
            print(f"  Go Toks: {go_tokens}")
            print(f"  HF Toks: {hf_tokens}")
            mismatches += 1
        else:
            # print(f"OK: '{text}'")
            pass
            
    if mismatches == 0:
        print("\nSUCCESS: All tokenizer cases matched!")
    else:
        print(f"\nFAILURE: {mismatches} mismatches found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
