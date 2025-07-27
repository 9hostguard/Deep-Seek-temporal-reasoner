from core.decomposition import decompose

def test_decompose():
    prompt = "I learned Python yesterday, I am coding today, and I will deploy tomorrow."
    parts = decompose(prompt)
    assert set(parts.keys()) == {"past", "present", "future"}
    assert "learned" in parts["past"] or "yesterday" in parts["past"]
    assert "coding" in parts["present"] or "today" in parts["present"]  
    assert "deploy" in parts["future"] or "tomorrow" in parts["future"]

if __name__ == "__main__":
    test_decompose()
    print("✅ Basic decomposition test passed!")

