"""
Monitoring's schema.
"""
from pydantic import BaseModel, field_validator


@field_validator("window_size")
def prevent_zero(_, value: int):
    """
    A function that will validate the parameter value for the
    'window_size' feature.

    Args:
        _ (str): the parameter's name (ignored).
        value (int): the given parameter value for that feature.

    Raises:
        ValueError: raises an error if the value is zero.

    Returns:
        int: the parameter's value.
    """
    if value == 0:
        raise ValueError("Ensure this value is not 0.")
    return value


class Monitoring(BaseModel):
    """
    Monitoring schema.

    window_size - The window size. Defaults to 300.
    """

    window_size: int = 300
