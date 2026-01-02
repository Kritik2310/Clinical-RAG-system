from clinical_rag.pipeline import detect_bias

def test_detects_age_bias():
    text = "Dementia is an inevitable part of aging."
    flags = detect_bias(text)
    assert "inevitable" in flags

def test_allows_safe_language():
    text = "Age is a risk factor, but dementia is not inevitable."
    flags = detect_bias(text)
    assert not flags
