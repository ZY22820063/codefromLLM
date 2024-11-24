from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

# 加载输入 CSV 文件，包含 SMILES 字符串
input_file = "list-A.csv"  # 替换为你的文件路径
df = pd.read_csv(input_file)

# 为每个分子的指纹准备一个字典
molecule_fp_dict = {}
error_indices = []  # 记录发生错误的序号

# 处理每个 SMILES 字符串
for index, row in df.iterrows():
    smiles = row['smiles']
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"跳过无效的 SMILES: {smiles}, 序号: {index}")
            error_indices.append(index)
            continue
        fp = AllChem.GetMorganFingerprint(mol, radius=3)
        molecule_fp_dict[smiles] = fp.GetNonzeroElements()
    except Exception as e:
        print(f"处理 SMILES 序号 {index}, SMILES: {smiles} 时发生错误: {e}")
        error_indices.append(index)
        continue

# 找到所有指纹中存在的所有 bits 的并集
all_bits = set()
for fp in molecule_fp_dict.values():
    all_bits.update(fp.keys())

# 创建一个填充为零的 DataFrame
mff_df = pd.DataFrame(0, index=df['smiles'], columns=all_bits)

# 使用频率填充 DataFrame
for smiles, fp in molecule_fp_dict.items():
    for bit, freq in fp.items():
        mff_df.at[smiles, bit] = freq

# 保存 DataFrame 到 CSV 文件
output_file = "mff_a2.csv"  # 替换为你想要的输出文件路径
mff_df.to_csv(output_file)

# 打印出错误的序号
if error_indices:
    print(f"以下序号的 SMILES 发生了错误: {error_indices}")
else:
    print("所有 SMILES 均成功处理")
