from dataclasses import dataclass

@dataclass
class Component:
    """
    表示 Diamine 或 Dianhydride 的一个成分
    """
    smiles: str
    proportion: float

    def __init__(self, smiles, proportion):
        self.smiles = smiles
        self.proportion = proportion

    def to_dict(self):
        return {"smiles": self.smiles, "proportion": self.proportion}