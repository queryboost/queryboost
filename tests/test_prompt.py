import pytest

from queryboost.exceptions import QueryboostPromptError
from queryboost.utils.prompt import validate_prompt


class TestValidatePrompt:
    """Test suite for validate_prompt function."""

    def test_valid_prompt_with_single_column(self):
        """Test prompt with single column reference is valid."""
        prompt = "Analyze this text: {content}"
        column_names = ["content", "id", "timestamp"]

        result = validate_prompt(prompt, column_names)
        assert result is True

    def test_valid_prompt_with_multiple_columns(self):
        """Test prompt with multiple column references is valid."""
        prompt = "Compare {text1} with {text2} and consider {context}"
        column_names = ["text1", "text2", "context", "metadata"]

        result = validate_prompt(prompt, column_names)
        assert result is True

    def test_valid_prompt_with_repeated_column(self):
        """Test prompt with repeated column reference is valid."""
        prompt = "First: {data}, Second: {data}, Third: {other}"
        column_names = ["data", "other"]

        result = validate_prompt(prompt, column_names)
        assert result is True

    def test_prompt_without_column_reference_raises_error(self):
        """Test prompt without any column reference raises error."""
        prompt = "This is a plain text prompt without any references"
        column_names = ["content", "id"]

        with pytest.raises(QueryboostPromptError) as exc_info:
            validate_prompt(prompt, column_names)

        assert "at least one column reference" in str(exc_info.value)

    def test_prompt_with_nonexistent_column_raises_error(self):
        """Test prompt with non-existent column reference raises error."""
        prompt = "Process this {content} and {missing_column}"
        column_names = ["content", "id", "timestamp"]

        with pytest.raises(QueryboostPromptError) as exc_info:
            validate_prompt(prompt, column_names)

        assert "not found" in str(exc_info.value)
        assert "missing_column" in str(exc_info.value)

    def test_prompt_with_multiple_nonexistent_columns_raises_error(self):
        """Test prompt with multiple non-existent columns raises error."""
        prompt = "Use {valid} with {missing1} and {missing2}"
        column_names = ["valid", "other"]

        with pytest.raises(QueryboostPromptError) as exc_info:
            validate_prompt(prompt, column_names)

        error_message = str(exc_info.value)
        assert "not found" in error_message
        assert "missing1" in error_message or "missing2" in error_message

    def test_empty_prompt_raises_error(self):
        """Test empty prompt raises error."""
        prompt = ""
        column_names = ["content"]

        with pytest.raises(QueryboostPromptError) as exc_info:
            validate_prompt(prompt, column_names)

        assert "Prompt is required" in str(exc_info.value)

    def test_prompt_with_braces_but_no_valid_reference(self):
        """Test prompt with braces but no valid column names raises error."""
        prompt = "This has {invalid} references"
        column_names = ["content", "data"]

        with pytest.raises(QueryboostPromptError) as exc_info:
            validate_prompt(prompt, column_names)

        assert "not found" in str(exc_info.value)

    def test_prompt_with_nested_braces(self):
        """Test prompt with nested braces format."""
        prompt = "Format: {{content}} and {actual_column}"
        column_names = ["actual_column", "content"]

        result = validate_prompt(prompt, column_names)
        assert result is True

    def test_prompt_with_format_specifiers(self):
        """Test prompt with format specifiers."""
        prompt = "Value: {column:.2f} and {other:>10}"
        column_names = ["column", "other"]

        result = validate_prompt(prompt, column_names)
        assert result is True

    def test_prompt_with_all_columns_referenced(self):
        """Test prompt that references all available columns."""
        prompt = "A: {col1}, B: {col2}, C: {col3}"
        column_names = ["col1", "col2", "col3"]

        result = validate_prompt(prompt, column_names)
        assert result is True

    def test_prompt_with_subset_of_columns(self):
        """Test prompt that references only subset of available columns."""
        prompt = "Use only {col1} and {col2}"
        column_names = ["col1", "col2", "col3", "col4", "col5"]

        result = validate_prompt(prompt, column_names)
        assert result is True

    def test_prompt_with_underscores_and_numbers(self):
        """Test column names with underscores and numbers."""
        prompt = "Data: {column_1} and {column_2_test}"
        column_names = ["column_1", "column_2_test", "other"]

        result = validate_prompt(prompt, column_names)
        assert result is True

    def test_case_sensitive_column_names(self):
        """Test that column name matching is case-sensitive."""
        prompt = "Use {Content}"
        column_names = ["content", "data"]  # lowercase 'content'

        with pytest.raises(QueryboostPromptError) as exc_info:
            validate_prompt(prompt, column_names)

        assert "Content" in str(exc_info.value)
        assert "not found" in str(exc_info.value)
