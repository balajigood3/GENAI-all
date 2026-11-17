from app.engine import ResearchEngine

def test_split_text():
    chunks = ResearchEngine.split_text("word " * 500, chunk_words=100, overlap=10)
    assert len(chunks) > 1
    assert all(chunks)

def test_empty_index_returns_no_evidence(tmp_path):
    engine = ResearchEngine(str(tmp_path))
    result = engine.answer("What does this research say?")
    assert result["mode"] == "no_evidence"
    assert result["evidence"] == []

def test_query_validation(tmp_path):
    engine = ResearchEngine(str(tmp_path))
    try:
        engine.answer(" ")
        assert False, "Expected ValueError"
    except ValueError:
        assert True
