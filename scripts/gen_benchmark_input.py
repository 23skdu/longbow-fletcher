import json

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a sleepy dog.",
    "The weather is nice today.",
    "It is a beautiful day.",
    "Artificial intelligence is transforming the world.",
    "AI is changing everything.",
    "I love pizza.",
    "Pizza is my favorite food."
]

# Generate 1000 sentences (125 repetitions of 8)
large_dataset = sentences * 125

with open("benchmark_input_1k.json", "w") as f:
    json.dump(large_dataset, f)

print(f"Generated benchmark_input_1k.json with {len(large_dataset)} items.")
