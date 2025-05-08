from dataclasses import dataclass
import pandas as pd

@dataclass
class SampleSchema:
    SampleID: str
    injectionID: str
    Group: pd.Categorical

@dataclass
class HighPHSchema:
    injectionID: str


@dataclass
class LowPHSchema:
    injectionID: str
