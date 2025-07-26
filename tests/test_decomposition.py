"""Tests for the temporal decomposition functionality."""

from core.decomposition import decompose


def test_decompose_basic():
    """Test basic decomposition functionality."""
    prompt = "Test past, present, and future."
    parts = decompose(prompt)
    assert set(parts.keys()) == {"past", "present", "future"}


def test_decompose_past_tense():
    """Test decomposition with past tense content."""
    prompt = "I was working yesterday and had completed the task before."
    parts = decompose(prompt)
    assert "past" in parts
    assert parts["past"]  # Should not be empty


def test_decompose_future_tense():
    """Test decomposition with future tense content."""
    prompt = "I will work tomorrow and shall complete the task soon."
    parts = decompose(prompt)
    assert "future" in parts
    assert parts["future"]  # Should not be empty


def test_decompose_present_tense():
    """Test decomposition with present tense content."""
    prompt = "I am working now and currently completing the task."
    parts = decompose(prompt)
    assert "present" in parts
    assert parts["present"]  # Should not be empty


def test_decompose_mixed_tenses():
    """Test decomposition with mixed temporal content."""
    prompt = "I worked yesterday, am working today, and will work tomorrow."
    parts = decompose(prompt)
    assert all(key in parts for key in ["past", "present", "future"])


def test_decompose_empty_string():
    """Test decomposition with empty input."""
    parts = decompose("")
    assert set(parts.keys()) == {"past", "present", "future"}


def test_decompose_no_temporal_markers():
    """Test decomposition with no clear temporal markers."""
    prompt = "This is a simple sentence without temporal indicators."
    parts = decompose(prompt)
    assert set(parts.keys()) == {"past", "present", "future"}
    # Should default to present
    assert parts["present"]


class TestDecompositionEdgeCases:
    """Test edge cases for temporal decomposition."""

    def test_very_long_prompt(self):
        """Test with a very long prompt."""
        long_prompt = "This is a test. " * 100
        parts = decompose(long_prompt)
        assert set(parts.keys()) == {"past", "present", "future"}

    def test_prompt_with_punctuation(self):
        """Test with various punctuation marks."""
        prompt = "Was it working? Yes! It will work. Currently it works."
        parts = decompose(prompt)
        assert set(parts.keys()) == {"past", "present", "future"}

    def test_prompt_with_numbers_and_symbols(self):
        """Test with numbers and symbols."""
        prompt = "In 2020, it was working. Now in 2024, it is working. In 2030, it will work."
        parts = decompose(prompt)
        assert set(parts.keys()) == {"past", "present", "future"}
