from clinical_rag.pipeline import extract_evidence

def test_evidence_is_not_empty():
    dummy_chunks = [
        "Risk factors for dementia include age and lifestyle factors.",
        "This sentence is irrelevant."
    ]

    evidence = extract_evidence(dummy_chunks, [])
    assert len(evidence) > 0
