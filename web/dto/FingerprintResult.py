from dataclasses import dataclass

@dataclass
class FingerprintResult:
    """
    表示指纹处理的结果
    """
    smiles: str
    fingerprint: dict