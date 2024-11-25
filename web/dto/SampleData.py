from dataclasses import dataclass
from typing import List
from web.dto.Component import Component

@dataclass
class SampleData:
    """
    表示整体的样品数据
    """
    temperature: float
    thickness: float
    time: float
    wavelength: float
    diamines: List[Component]
    dianhydrides: List[Component]
    summed_fingerprints: dict

    def __init__(self, temperature, thickness, time, wavelength, diamines, dianhydrides):
        self.temperature = temperature
        self.thickness = thickness
        self.time = time
        self.wavelength = wavelength
        self.diamines = diamines
        self.dianhydrides = dianhydrides

    def to_dict(self):
        return {
            "temperature": self.temperature,
            "thickness": self.thickness,
            "time": self.time,
            "wavelength": self.wavelength,
            "diamines": [d.to_dict() for d in self.diamines],
            "dianhydrides": [d.to_dict() for d in self.dianhydrides],
            "summed_fingerprints": self.summed_fingerprints
        }