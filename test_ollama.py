import ollama

print("1. Pulling nomic-embed-text embedding model...")
ollama.pull("nomic-embed-text")
print("Embedding model downloaded successfully!")

print("\n2. Testing embedding generation...")
res = ollama.embeddings(model="nomic-embed-text", prompt="Change of control clause")
print(f"Generated vector with {len(res['embedding'])} dimensions.")

print("\nAll systems ready!")
