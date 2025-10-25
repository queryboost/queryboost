import string

from queryboost.exceptions import QueryboostPromptError


def validate_prompt(prompt: str, column_names: list[str]) -> bool:
    """Validate that the prompt is a string template with at least one column name reference.

    Args:
        prompt: The prompt to validate.
        column_names: The column names in the data.

    Returns:
        True if the prompt is valid.

    Raises:
        QueryboostPromptError: If the prompt is invalid.

    :meta private:
    """

    if not prompt:
        raise QueryboostPromptError("Prompt is required.")

    # Extract column names from the prompt
    prompt_column_names = [column_name for _, column_name, _, _ in string.Formatter().parse(prompt) if column_name]

    # Check if the prompt has at least one column reference
    if not prompt_column_names:
        raise QueryboostPromptError("Prompt requires at least one column reference.")

    # Check if all column names in the prompt are present in the data
    elif extra_column_names := set(prompt_column_names) - set(column_names):
        raise QueryboostPromptError(f"Column reference(s) not found in data: {', '.join(extra_column_names)}.")

    return True
