import chromadb
import ollama

# Sample M&A Contract with real legal clauses
SAMPLE_CONTRACT = """CREDIT AND ACQUISITION AGREEMENT - CONFIDENTIAL

SECTION 4.1: CHANGE OF CONTROL
Upon the occurrence of a Change of Control event, the Borrower shall provide written notice to the Administrative Agent within three (3) business days. The Majority Lenders shall have the immediate right to demand full prepayment of all outstanding Principal Amounts, accrued interest, and a mandatory prepayment penalty fee equal to 3.5% of the aggregate principal balance.

SECTION 7.4: NON-COMPETE RESTRICTIONS
For a period of thirty-six (36) months following the Closing Date, the Target Company Founders and Key Executives shall not directly or indirectly engage in, invest in, or consult for any business operating in automated legal analysis, corporate diligence, or contract extraction within North America and the European Union.

SECTION 9.2: BREAK-UP FEE AND LIQUIDATED DAMAGES
If the Seller terminates this Agreement to pursue an Alternative Transaction or Superior Proposal, Seller shall immediately remit to Buyer a cash termination break-up fee of $15,000,000 within forty-eight (48) hours of written notice of termination.

SECTION 12.1: INDEMNIFICATION AND SURVIVAL
All representations and warranties shall survive the Closing Date for a duration of 24 months. The Target Company agrees to indemnify Buyer against all undisclosed tax liabilities, pending IP litigation, and environmental claims exceeding a basket threshold of $200,000, up to a maximum liability cap equal to 20% of Enterprise Value."""

def get_embedding(text: str):
    """Converts a clause into a search vector using nomic-embed-text."""
    res = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return res["embedding"]

def run():
    print("1. Initializing ChromaDB vector store...")
    client = chromadb.PersistentClient(path="diligence_db")
    collection = client.get_or_create_collection(name="contract_clauses")

    # Split into sections by blank lines
    sections = [s.strip() for s in SAMPLE_CONTRACT.strip().split("\n\n") if s.strip()]

    print(f"2. Found {len(sections)} sections. Embedding into ChromaDB...")
    for idx, sec in enumerate(sections):
        first_line = sec.split("\n")[0]
        embedding = get_embedding(sec)
        
        collection.upsert(
            ids=[f"clause_{idx}"],
            embeddings=[embedding],
            documents=[sec],
            metadatas=[{"source": "Credit_Agreement_2026.pdf", "section": first_line}]
        )
        print(f"   Indexed: {first_line}")

    print("\nSuccess! Vector database created in folder 'diligence_db'.")

if __name__ == "__main__":
    run()
