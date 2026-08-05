"""Unit tests for `app.processing.cleaner`."""
from app.processing.cleaner import clean_text


class TestCleanText:
  def test_normalizes_internal_whitespace(self) -> None:
    raw = "Hello   world\t\there"
    assert clean_text(raw) == "Hello world here"

  def test_removes_empty_lines(self) -> None:
    raw = "Line one\n\n\nLine two"
    assert clean_text(raw) == "Line one\n\nLine two"

  def test_preserves_paragraph_boundaries(self) -> None:
    raw = "Paragraph one.\n\nParagraph two."
    result = clean_text(raw)
    assert result == "Paragraph one.\n\nParagraph two."

  def test_collapses_multiline_paragraph(self) -> None:
    raw = "Line one\nLine two"
    assert clean_text(raw) == "Line one\nLine two"

  def test_empty_input_returns_empty_string(self) -> None:
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""
