# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:16:28 2024

@author: 15297
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from catboost import CatBoostRegressor  # 导入CatBoost回归器
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 加载训练集和测试集
train_data = pd.read_csv('train12.csv')  # 替换为正确的文件路径
test_data = pd.read_csv('test12.csv')    # 替换为正确的文件路径

# 选定的特征
selected_features = ['Thickness', 'Total Imidization Time', 'g2434951923', 
                     'a864674487', 'Maximum Imidization Temperature', 'g951226070', 
                     'g4290259127', 'g3217380708', 'g2968968094', 'a3217380708']

# 分离特征和目标
X_train = train_data[selected_features]
y_train = train_data.iloc[:, 0]
X_test = test_data[selected_features]
y_test = test_data.iloc[:, 0]

# 参数网格
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4],
    'depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'iterations': [100, 200, 300, 400],
    'l2_leaf_reg': [1, 3, 5, 10]
}

# 初始化CatBoost回归器
#catboost = CatBoostRegressor(verbose=0, random_state=42)
catboost = CatBoostRegressor(loss_function='RMSE', verbose=0, random_state=42)

# 修改KFold设置，添加shuffle和random_state
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 网格搜索，使用多个评估指标
grid_search = GridSearchCV(
    catboost,
    param_grid,
    cv=kf,
    scoring={'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'},
    n_jobs=-1,
    return_train_score=True,  # 获取训练集评分
    refit='rmse'  # 使用RMSE进行模型拟合
)
grid_search.fit(X_train, y_train)

# 结果输出
results = []
for i in range(len(grid_search.cv_results_['params'])):
    params = grid_search.cv_results_['params'][i]
    mean_train_rmse = -grid_search.cv_results_['mean_train_rmse'][i]
    mean_train_r2 = grid_search.cv_results_['mean_train_r2'][i]
    mean_val_rmse = -grid_search.cv_results_['mean_test_rmse'][i]
    mean_val_r2 = grid_search.cv_results_['mean_test_r2'][i]

    # 使用当前参数重新训练模型以获取测试集上的评分
    model = CatBoostRegressor(**params, verbose=0, random_state=42)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    # 添加到结果中
    results.append({
        'learning_rate': params['learning_rate'],
        'depth': params['depth'],
        'subsample': params.get('subsample'),
        'iterations': params['iterations'],
        'l2_leaf_reg': params['l2_leaf_reg'],
        'CV-10-train-RMSE': mean_train_rmse,
        'CV-10-train-R2': mean_train_r2,
        'CV-10-validation-RMSE': mean_val_rmse,
        'CV-10-validation-R2': mean_val_r2,
        'test-RMSE': test_rmse,
        'test-R2': test_r2
    })

# 保存结果到CSV文件
results_df = pd.DataFrame(results)
results_df.to_csv('micatboost_parameter-tt12-10-promote.csv', index=False)

print("Parameter optimization results saved to 'catboost_parameter_optimization.csv'.")
