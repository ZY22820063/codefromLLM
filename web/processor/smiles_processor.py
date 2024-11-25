from rdkit import Chem
from rdkit.Chem import AllChem

from web.dto.Component import Component
from web.dto.FingerprintResult import FingerprintResult
from web.dto.SampleData import SampleData


def process_sample_data(sample: SampleData):
    """
    处理 SampleData 数据，返回指纹结果
    """
    fingerprints = []

    for diamine in sample.diamines:
        fingerprints.append(
            process_fingerprint_result(diamine, 'a')
        )

    for dianhydride in sample.dianhydrides:
        fingerprints.append(
            process_fingerprint_result(dianhydride, 'g')
        )

    # 累加指纹结果
    summed_fingerprints = {}
    for fp_result in fingerprints:
        for key, value in fp_result.fingerprint.items():
            if key in summed_fingerprints:
                summed_fingerprints[key] += value
            else:
                summed_fingerprints[key] = value

    sample.summed_fingerprints = summed_fingerprints


def process_fingerprint_result(component: Component, prefix: str):
    try:
        mol = Chem.MolFromSmiles(component.smiles)
        if mol is None:
            print(f"无效的 SMILES: {component.smiles}")
            return FingerprintResult(smiles=component.smiles, fingerprint={})
        fp = AllChem.GetMorganFingerprint(mol, radius=3)
        fp_results = fp.GetNonzeroElements()
        # 使用字典推导式统一修改所有 value
        scaled_fp_results = {prefix + str(k): v * component.proportion for k, v in fp_results.items()}
        return FingerprintResult(smiles=component.smiles, fingerprint=scaled_fp_results)
    except Exception as e:
        print(f"处理 SMILES {component.smiles} 时发生错误: {e}")
        return FingerprintResult(smiles=component.smiles, fingerprint={})
