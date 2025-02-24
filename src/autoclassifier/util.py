from dataclasses import dataclass
from typing import Optional

@dataclass
class Labeled:
    index: int
    label: None | int
