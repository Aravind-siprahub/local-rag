"""Unit tests for `app.processing.chunker`."""
import pytest

from app.processing.chunker import chunk_text


class TestChunkText:
  def test_returns_empty_for_blank_text(self) -> None:
    assert chunk_text("", 100, 10) == []
    assert chunk_text("   ", 100, 10) == []

  def test_single_chunk_when_text_fits(self) -> None:
    text = "Short text."
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].content == "Short text."
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)

  def test_splits_long_text_with_overlap(self) -> None:
    text = "a" * 250
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert all(chunk.content for chunk in chunks)

  def test_preserves_chunk_order(self) -> None:
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    indices = [chunk.chunk_index for chunk in chunks]
    assert indices == list(range(len(chunks)))

  def test_char_metadata_covers_original_text(self) -> None:
    text = "Hello world. " * 20
    chunks = chunk_text(text, chunk_size=40, overlap=5)
    for chunk in chunks:
      assert text[chunk.char_start:chunk.char_end] == chunk.content

  def test_prefers_paragraph_breaks(self) -> None:
    text = "First paragraph.\n\nSecond paragraph is here."
    chunks = chunk_text(text, chunk_size=25, overlap=5)
    assert any("First paragraph." in chunk.content for chunk in chunks)

  def test_rejects_invalid_chunk_size(self) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
      chunk_text("text", chunk_size=0, overlap=0)

  def test_rejects_overlap_gte_chunk_size(self) -> None:
    with pytest.raises(ValueError, match="overlap"):
      chunk_text("text", chunk_size=10, overlap=10)
