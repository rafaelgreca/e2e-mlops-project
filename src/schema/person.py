"""
Creates a person schema with Pydantic's base model, which will be used to
validate the parameters value when passed to the API.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator


@field_validator("Age", "Height", "Weight", "FCVC", "CH2O")
def prevent_zero(_, value: int):
    """
    A function that will validate the parameter value for the
    'Age', 'Height', 'Weight', 'CH2O', and 'FCVC' features.

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


class Person(BaseModel):
    """
    Person schema.

    Age - The person's age.
    Height - The person's height (in meters).
    Weight - The person's weight (in kilos).
    Gender - The person's gender.
    CALC - The person's consumption of alcohol (CALC).
    FAVC - The person's frequent consumption of high caloric food (FAVC).
    family_history_with_overweight - Whether the person's family has history
        with overweight.
    MTRANS - Transportation used (MTRANS).
    FCVC - Frequency of Consumption of Vegetables (FCVC).
    FAF - Physical activity frequency (FAF).
    NCP - The person's number of main meals (NCP).
    CH20 - The person's consumption of water daily (CH2O).
    TUE - Time using technology devices (TUE).
    CAEC - The person's consumption of food between meals (CAEC).
    SCC - Whether the person monitor they calories consumption (SCC).
    """

    Age: float = Field(ge=0, le=100)
    Height: float = Field(ge=0.0, le=2.5)
    Weight: float = Field(ge=0, le=400)
    Gender: str = Literal["Male", "Female"]
    CALC: str = Literal["Frequently", "Sometimes", "no"]
    FAVC: str = Literal["yes", "no"]
    family_history_with_overweight: str = Literal["yes", "no"]
    MTRANS: str = Literal[
        "Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"
    ]
    FCVC: float = Field(ge=0, le=5)
    NCP: float = Field(ge=0, le=4)
    CH2O: float = Field(ge=1, le=3)
    FAF: float = Field(ge=0, le=3)
    TUE: int = Field(ge=0, le=2)
    CAEC: str = Literal["Frequently", "Sometimes", "Always", "no"]
    SCC: str = Literal["yes", "no"]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 24.443011,
                    "Height": 1.699998,
                    "Weight": 81.66995,
                    "Gender": "Male",
                    "family_history_with_overweight": "yes",
                    "CALC": "Sometimes",
                    "MTRANS": "Public_Transportation",
                    "FAVC": "yes",
                    "FCVC": 2,
                    "NCP": 2.983297,
                    "CH2O": 2.763573,
                    "FAF": 0,
                    "TUE": 1,
                    "CAEC": "Sometimes",
                    "SCC": "no",
                }
            ]
        }
    }
