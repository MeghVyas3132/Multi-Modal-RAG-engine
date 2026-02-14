"""
Unit tests for upload filename sanitization.

Covers path traversal prevention, null bytes, dangerous characters,
and edge cases in _sanitize_filename().
"""
import os
import sys

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.api_gateway.endpoints.upload import _sanitize_filename


class TestSanitizeFilename:
    """Test the _sanitize_filename helper."""

    def test_normal_filename(self):
        assert _sanitize_filename("report.pdf") == "report.pdf"

    def test_path_traversal_relative(self):
        result = _sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
        assert result == "passwd"

    def test_path_traversal_absolute(self):
        result = _sanitize_filename("/etc/shadow")
        assert result == "shadow"
        assert "/" not in result

    def test_path_traversal_backslash(self):
        result = _sanitize_filename("..\\..\\windows\\system32\\cmd.exe")
        # os.path.basename handles platform-specific separators
        assert ".." not in result
        assert "\\" not in result

    def test_null_bytes_stripped(self):
        result = _sanitize_filename("file\x00.pdf")
        assert "\x00" not in result
        assert result == "file.pdf"

    def test_special_characters_replaced(self):
        result = _sanitize_filename("my<file>:name|?.pdf")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "|" not in result
        assert "?" not in result

    def test_empty_filename_fallback(self):
        assert _sanitize_filename("") == "upload"
        assert _sanitize_filename("...") == "upload"
        assert _sanitize_filename("   ") == "upload"

    def test_only_dots(self):
        result = _sanitize_filename(".....")
        assert result == "upload"

    def test_spaces_preserved(self):
        result = _sanitize_filename("my file name.pdf")
        assert "my file name.pdf" == result

    def test_unicode_filename(self):
        result = _sanitize_filename("日本語ファイル.pdf")
        # Should not crash, filename may be sanitized but should be non-empty
        assert len(result) > 0

    def test_double_dots_collapsed(self):
        result = _sanitize_filename("file..hidden..pdf")
        assert ".." not in result

    def test_dot_slash_traversal(self):
        result = _sanitize_filename("./hidden/../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
        assert result == "passwd"

    def test_long_filename(self):
        """Very long filenames should not crash."""
        name = "a" * 500 + ".pdf"
        result = _sanitize_filename(name)
        assert result.endswith(".pdf")
