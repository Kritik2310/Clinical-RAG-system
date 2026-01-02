from clinical_rag.pipeline import extract_evidence

def test_fallback_when_no_evidence():
    chunks = ["This text talks about weather."]
    evidence = extract_evidence(chunks, [])

    assert evidence == []
