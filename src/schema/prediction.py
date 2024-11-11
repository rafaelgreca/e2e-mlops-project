from typing import List, Literal

from pydantic import BaseModel

class Prediction(BaseModel):
    """
    Model's prediction schema.

    predictions - A list containing the predicted class for each data sample.
    """
    predictions: List[
        Literal[
            "Insufficient Weight",
            "Normal Weight",
            "Overweight Level I",
            "Overweight Level II",
            "Obesity Type I",
            "Obesity Type II",
            "Obesity Type III"
        ]
    ]
