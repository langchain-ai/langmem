"""Tests for schema compatibility between hot path and cold path memory operations.

Verifies fixes for:
- Issue #138: MemoryStoreManager crashes when reading memories written by
  create_manage_memory_tool due to missing "kind" key.
- Issue #140: create_search_memory_tool returns nested/inconsistent format
  for memories written by MemoryStoreManager.
"""

import pytest

from langmem.knowledge.extraction import _parse_store_value
from langmem.knowledge.tools import _normalize_memory_value


class TestParseStoreValue:
    """Test _parse_store_value handles both hot path and cold path formats."""

    def test_cold_path_format(self):
        """Cold path (MemoryStoreManager) writes {"kind": "Memory", "content": {...}}."""
        value = {"kind": "Memory", "content": {"content": "User likes Python"}}
        kind, content = _parse_store_value(value)
        assert kind == "Memory"
        assert content == {"content": "User likes Python"}

    def test_cold_path_custom_schema(self):
        """Cold path with custom Pydantic schema."""
        value = {"kind": "UserProfile", "content": {"name": "Alice", "age": 30}}
        kind, content = _parse_store_value(value)
        assert kind == "UserProfile"
        assert content == {"name": "Alice", "age": 30}

    def test_hot_path_string_content(self):
        """Hot path (manage_memory_tool) writes {"content": "bare string"} for str schema."""
        value = {"content": "User is a data engineer"}
        kind, content = _parse_store_value(value)
        assert kind == "Memory"
        assert content == {"content": "User is a data engineer"}

    def test_hot_path_dict_content(self):
        """Hot path with custom Pydantic schema serialized to dict."""
        value = {"content": {"name": "Alice", "age": 30}}
        kind, content = _parse_store_value(value)
        assert kind == "Memory"
        assert content == {"name": "Alice", "age": 30}

    def test_hot_path_numeric_content(self):
        """Hot path with numeric content (edge case)."""
        value = {"content": 42}
        kind, content = _parse_store_value(value)
        assert kind == "Memory"
        assert content == {"content": 42}

    def test_empty_dict(self):
        """Edge case: empty dict."""
        value = {}
        kind, content = _parse_store_value(value)
        assert kind == "Memory"
        assert content == {}

    def test_unknown_keys(self):
        """Edge case: dict with unknown keys and no content."""
        value = {"foo": "bar"}
        kind, content = _parse_store_value(value)
        assert kind == "Memory"
        assert content == {"foo": "bar"}


class TestNormalizeMemoryValue:
    """Test _normalize_memory_value unwraps cold path envelope for search results."""

    def test_cold_path_envelope(self):
        """Cold path envelope should be unwrapped."""
        value = {"kind": "Memory", "content": {"content": "User likes Python"}}
        result = _normalize_memory_value(value)
        assert result == {"content": "User likes Python"}

    def test_cold_path_custom_schema(self):
        """Custom schema envelope should be unwrapped."""
        value = {"kind": "UserProfile", "content": {"name": "Alice", "age": 30}}
        result = _normalize_memory_value(value)
        assert result == {"name": "Alice", "age": 30}

    def test_hot_path_passthrough(self):
        """Hot path format should pass through unchanged."""
        value = {"content": "User is a data engineer"}
        result = _normalize_memory_value(value)
        assert result == {"content": "User is a data engineer"}

    def test_non_dict_passthrough(self):
        """Non-dict values should pass through unchanged."""
        assert _normalize_memory_value("hello") == "hello"
        assert _normalize_memory_value(42) == 42
        assert _normalize_memory_value(None) is None

    def test_empty_dict(self):
        """Empty dict should pass through."""
        assert _normalize_memory_value({}) == {}

    def test_partial_envelope(self):
        """Dict with only 'kind' but no 'content' should pass through."""
        value = {"kind": "Memory", "data": "something"}
        result = _normalize_memory_value(value)
        assert result == {"kind": "Memory", "data": "something"}
