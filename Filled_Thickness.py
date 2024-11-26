# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 20:52:57 2024

@author: 15297
"""

import pandas as pd
import numpy as np

# 示例数据
data = {
    'Thickness (µm)': [40, 40, 40, 50, np.nan, 160, 200, np.nan, 70, 60, np.nan, 600, np.nan, 42.5, 27.5, np.nan, 70, 50, np.nan, 20, 55, 150, 18, np.nan, 40, np.nan, 20, 37.5, 35, np.nan, 34, np.nan, 16, np.nan, 60, 12.5, np.nan, 29, 50, 40, np.nan, 50, np.nan, 74, 50, np.nan]
}

df = pd.DataFrame(data)

# 使用线性插值法填补缺失值
df['Thickness (µm)'] = df['Thickness (µm)'].interpolate(method='linear')

# 显示填补后的数据
print(df)
