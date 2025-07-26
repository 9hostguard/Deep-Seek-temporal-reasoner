from core.decomposition import decompose


def test_decompose():
    prompt = "Test past, present, and future."
    parts = decompose(prompt)
    assert set(parts.keys()) == {"past", "present", "future"}
