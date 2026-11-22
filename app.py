import chainlit as cl
import chromadb
import ollama
from pypdf import PdfReader

def get_embedding(text: str):
    res = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return res["embedding"]

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

@cl.on_chat_start
async def on_start():
    await cl.Message(
        content="### ⚖️ Universal AI Document Copilot\n"
                "You can ask questions about pre-loaded documents, or **attach any PDF file** using the paperclip icon (📎) below to index and analyze it live!"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    client = chromadb.PersistentClient(path="diligence_db")
    collection = client.get_or_create_collection(name="contract_clauses")

    # 1. Check if user attached a PDF file
    if message.elements:
        for element in message.elements:
            if element.mime == "application/pdf" or element.name.endswith(".pdf"):
                status_msg = await cl.Message(content=f"⏳ Processing and indexing `{element.name}`...").send()
                
                # Read PDF pages
                reader = PdfReader(element.path)
                full_text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        full_text += extracted + "\n"

                # Split into chunks and embed into ChromaDB
                chunks = chunk_text(full_text)
                for idx, chunk in enumerate(chunks):
                    vec = get_embedding(chunk)
                    collection.add(
                        ids=[f"{element.name}_chunk_{idx}"],
                        embeddings=[vec],
                        documents=[chunk],
                        metadatas=[{"section": f"Chunk {idx+1}", "source": element.name}]
                    )

                status_msg.content = f"✅ Successfully indexed `{element.name}` into ChromaDB! Ask any question about it."
                await status_msg.update()
                return

    # 2. Answer question using RAG
    question = message.content
    query_vector = get_embedding(question)
    results = collection.query(query_embeddings=[query_vector], n_results=1)

    if not results["documents"] or not results["documents"][0]:
        await cl.Message(content="No relevant documents found. Please attach a PDF or add documents first.").send()
        return

    retrieved_doc = results["documents"][0][0]
    retrieved_meta = results["metadatas"][0][0]
    section_name = retrieved_meta.get("section", "Document Excerpt")
    source_file = retrieved_meta.get("source", "Uploaded Document")

    # Display evidence card
    elements = [
        cl.Text(
            name=f"📄 {section_name} ({source_file})",
            content=retrieved_doc,
            display="inline"
        )
    ]

    msg = cl.Message(content="", elements=elements)
    await msg.send()

    system_prompt = (
        "You are an expert Document Intelligence Analyst. "
        "Answer the question using strictly the provided document context. "
        "If the answer is not in the text, clearly state that."
    )

    user_prompt = f"""Document Excerpt:
{retrieved_doc}

Question: {question}

Provide your analysis:
- **Finding:** [Direct answer]
- **Source Ref:** {section_name} from {source_file}
- **Key Takeaways / Risks:** [Summary]"""

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )

    for chunk in response:
        content = chunk["message"]["content"]
        await msg.stream_token(content)

    await msg.update()


