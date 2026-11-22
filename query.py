import chromadb
import ollama

def get_embedding(text: str):
    res = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return res["embedding"]

def ask_diligence_copilot(question: str):
    print(f"\nUser Question: {question}\n")
    print("1. Searching vector database for relevant clauses...")

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path="diligence_db")
    collection = client.get_collection(name="contract_clauses")

    # RETRIEVAL: Find the best matching clause
    query_vector = get_embedding(question)
    results = collection.query(query_embeddings=[query_vector], n_results=1)

    retrieved_doc = results["documents"][0][0]
    retrieved_meta = results["metadatas"][0][0]

    print(f"2. Retrieved: [{retrieved_meta['section']}] from {retrieved_meta['source']}")

    # AUGMENTATION & GENERATION: Prompt Llama 3.1
    system_prompt = (
        "You are an expert M&A Legal Due Diligence Analyst. "
        "Answer the user's question using ONLY the provided contract excerpt. "
        "Cite the section and highlight any financial penalties or restrictions."
    )

    user_prompt = f"""Contract Excerpt:
{retrieved_doc}

Question: {question}

Provide your findings in 3 concise bullet points:
- Finding:
- Contract Section:
- Risk / Financial Impact:"""

    print("3. Generating analysis with Llama 3.1...\n")
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    print("=" * 60)
    print("DILIGENCE REPORT")
    print("=" * 60)
    print(response["message"]["content"])
    print("=" * 60)

if __name__ == "__main__":
    # Test question:
    ask_diligence_copilot("What is the penalty if the seller backs out to accept a better offer?")
