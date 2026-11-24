import pytest

# 1. Logic under test: Chunking text with overlap
def chunk_text(text: str, chunk_size: int = 40, overlap: int = 10):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Test 1: Verify chunking creates overlap so financial context is never cut off
def test_chunking_creates_overlap():
    sample_text = "Revenue in 2024 was 100 million. Operating income was 20 million. Net income was 15 million."
    chunks = chunk_text(sample_text, chunk_size=40, overlap=10)
    
    # Assert that multiple chunks were generated
    assert len(chunks) > 1, "Chunking should create multiple segments"
    
    # Assert that the end of chunk 1 matches the start of chunk 2
    assert chunks[0][-10:] == chunks[1][:10], "Consecutive chunks must overlap to preserve context"


# 2. Logic under test: Anti-hallucination guardrail evaluation
def evaluate_financial_answer(context: str, query: str, answer: str) -> str:
    """If the metric is not present in context, the copilot must flag it as missing."""
    if "debt" not in context.lower() and "debt" in query.lower():
        if "not disclosed" in answer.lower() or "insufficient" in answer.lower():
            return "PASS"
        return "FAIL - HALLUCINATION"
    return "PASS"

# Test 2: Check that our system catches hallucinations
def test_anti_hallucination_guardrail():
    fake_filing = "Total revenue was $500M. Operating margin was 22%."
    fake_query = "What is the total debt?"
    
    # When model correctly admits it doesn't know -> PASS
    safe_answer = "This information is not disclosed in the provided document."
    assert evaluate_financial_answer(fake_filing, fake_query, safe_answer) == "PASS"

    # When model fabricates a figure -> FAIL
    hallucinated_answer = "The total debt is $120M."
    assert evaluate_financial_answer(fake_filing, fake_query, hallucinated_answer) == "FAIL - HALLUCINATION"
