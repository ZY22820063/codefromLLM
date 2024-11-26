# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:42:32 2024

@author: 15297
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 加载已经确定的训练集和待预测的数据集
train_data = pd.read_csv('train.csv')  # 包含已经确定的训练集
predict_data = pd.read_csv('HT-pred.csv')  # 包含需要预测的数据集

# 分离训练集的特征和目标
X_train = train_data.iloc[:, 3:].values  # 假设前三列是目标，后面是特征
y_train_tr = train_data['tr'].values
y_train_tg = train_data['tg'].values
y_train_ts = train_data['ts'].values

# 待预测数据集的特征
X_predict = predict_data.iloc[:, 3:].values  # 只取特征列

# 已确定的模型超参数
best_params = {
    'iterations': 1054,
    'depth': 6,
    'learning_rate': 0.078912001,
    'l2_leaf_reg': 6.370825503,
    'random_strength': 2.84965955,
    'bagging_temperature': 2.274178198,
    'border_count': 156,
    'verbose': 0
}

# 训练模型 tr
model_tr = CatBoostRegressor(**best_params, random_state=42)
model_tr.fit(X_train, y_train_tr)
y_pred_tr = model_tr.predict(X_predict)  # 预测目标 tr

# 训练模型 tg
model_tg = CatBoostRegressor(**best_params, random_state=42)
model_tg.fit(X_train, y_train_tg)
y_pred_tg = model_tg.predict(X_predict)  # 预测目标 tg

# 训练模型 ts
model_ts = CatBoostRegressor(**best_params, random_state=42)
model_ts.fit(X_train, y_train_ts)
y_pred_ts = model_ts.predict(X_predict)  # 预测目标 ts

# 将预测结果保存为 DataFrame
predictions = pd.DataFrame({
    'y_pred_tr': y_pred_tr,
    'y_pred_tg': y_pred_tg,
    'y_pred_ts': y_pred_ts
})

# 保存预测结果到 CSV 文件
predictions.to_csv('predictions_for_hT.csv', index=False)

# 打印说明
print("Predictions for the new data have been saved to 'predictions_for_new_data.csv'.")