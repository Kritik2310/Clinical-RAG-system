from clinical_rag.pipeline import cosine_similarity,model,embeddings,chunks

def test_retrieval_mentions_risk():
    ques="what are the risk factors for cognitive decline?"
    
    q_emb = model.encode(ques)

    scores = []
    for i, emb in enumerate(embeddings):
        scores.append((i, cosine_similarity(q_emb, emb)))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_chunk = chunks[scores[0][0]].lower()

    assert any(
        phrase in top_chunk
        for phrase in ["risk factor", "associated with", "increased risk","include","increase risk"]
    )