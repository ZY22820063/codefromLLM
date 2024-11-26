# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:51:10 2024

@author: 15297
"""

import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# 加载固定的训练集和测试集
train_data = pd.read_csv('train19.csv')
test_data = pd.read_csv('test19.csv')

X_train = train_data.iloc[:, 3:].values  # 假设前三列是目标，后面是特征
y_train_tr = train_data['tr'].values
y_train_tg = train_data['tg'].values
y_train_ts = train_data['ts'].values

X_test = test_data.iloc[:, 3:].values
y_test_tr = test_data['tr'].values
y_test_tg = test_data['tg'].values
y_test_ts = test_data['ts'].values

# 设置10折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 用于存储结果的列表
results = []
train_predictions = []
test_predictions = []

# 定义目标函数，用于贝叶斯优化
def objective(trial):
    # 选择一组超参数
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 100),
        'random_strength': trial.suggest_float('random_strength', 1e-5, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'verbose': 0
    }

    # 初始化性能指标
    avg_rmse_tr_train = avg_rmse_tr_test = 0
    avg_r2_tr_train = avg_r2_tr_test = 0

    avg_rmse_tg_train = avg_rmse_tg_test = 0
    avg_r2_tg_train = avg_r2_tg_test = 0

    avg_rmse_ts_train = avg_rmse_ts_test = 0
    avg_r2_ts_train = avg_r2_ts_test = 0

    fold_train_predictions = []
    fold_test_predictions = []

    # 进行10折交叉验证
    for train_idx, test_idx in kf.split(X_train):
        X_cv_train, X_cv_test = X_train[train_idx], X_train[test_idx]
        y_cv_train_tr, y_cv_test_tr = y_train_tr[train_idx], y_train_tr[test_idx]
        y_cv_train_tg, y_cv_test_tg = y_train_tg[train_idx], y_train_tg[test_idx]
        y_cv_train_ts, y_cv_test_ts = y_train_ts[train_idx], y_train_ts[test_idx]

        # 训练和评估 tr
        model_tr = CatBoostRegressor(**params, random_state=42)
        model_tr.fit(X_cv_train, y_cv_train_tr)
        y_pred_tr_train = model_tr.predict(X_cv_train)
        y_pred_tr_test = model_tr.predict(X_cv_test)
        avg_rmse_tr_train += np.sqrt(mean_squared_error(y_cv_train_tr, y_pred_tr_train))
        avg_rmse_tr_test += np.sqrt(mean_squared_error(y_cv_test_tr, y_pred_tr_test))
        avg_r2_tr_train += r2_score(y_cv_train_tr, y_pred_tr_train)
        avg_r2_tr_test += r2_score(y_cv_test_tr, y_pred_tr_test)

        # 训练和评估 tg
        model_tg = CatBoostRegressor(**params, random_state=42)
        model_tg.fit(X_cv_train, y_cv_train_tg)
        y_pred_tg_train = model_tg.predict(X_cv_train)
        y_pred_tg_test = model_tg.predict(X_cv_test)
        avg_rmse_tg_train += np.sqrt(mean_squared_error(y_cv_train_tg, y_pred_tg_train))
        avg_rmse_tg_test += np.sqrt(mean_squared_error(y_cv_test_tg, y_pred_tg_test))
        avg_r2_tg_train += r2_score(y_cv_train_tg, y_pred_tg_train)
        avg_r2_tg_test += r2_score(y_cv_test_tg, y_pred_tg_test)

        # 训练和评估 ts
        model_ts = CatBoostRegressor(**params, random_state=42)
        model_ts.fit(X_cv_train, y_cv_train_ts)
        y_pred_ts_train = model_ts.predict(X_cv_train)
        y_pred_ts_test = model_ts.predict(X_cv_test)
        avg_rmse_ts_train += np.sqrt(mean_squared_error(y_cv_train_ts, y_pred_ts_train))
        avg_rmse_ts_test += np.sqrt(mean_squared_error(y_cv_test_ts, y_pred_ts_test))
        avg_r2_ts_train += r2_score(y_cv_train_ts, y_pred_ts_train)
        avg_r2_ts_test += r2_score(y_cv_test_ts, y_pred_ts_test)

        # 保存预测结果
        fold_train_predictions.append({
            'y_true_tr': y_cv_train_tr, 'y_pred_tr': y_pred_tr_train,
            'y_true_tg': y_cv_train_tg, 'y_pred_tg': y_pred_tg_train,
            'y_true_ts': y_cv_train_ts, 'y_pred_ts': y_pred_ts_train
        })
        fold_test_predictions.append({
            'y_true_tr': y_cv_test_tr, 'y_pred_tr': y_pred_tr_test,
            'y_true_tg': y_cv_test_tg, 'y_pred_tg': y_pred_tg_test,
            'y_true_ts': y_cv_test_ts, 'y_pred_ts': y_pred_ts_test
        })

    # 计算10折的平均值
    avg_rmse_tr_train /= 10
    avg_rmse_tr_test /= 10
    avg_r2_tr_train /= 10
    avg_r2_tr_test /= 10

    avg_rmse_tg_train /= 10
    avg_rmse_tg_test /= 10
    avg_r2_tg_train /= 10
    avg_r2_tg_test /= 10

    avg_rmse_ts_train /= 10
    avg_rmse_ts_test /= 10
    avg_r2_ts_train /= 10
    avg_r2_ts_test /= 10

    # 保存预测值
    train_predictions.append(fold_train_predictions)
    test_predictions.append(fold_test_predictions)

    # 保存结果
    results.append({
        'trial': trial.number,
        'iterations': params['iterations'],
        'depth': params['depth'],
        'learning_rate': params['learning_rate'],
        'l2_leaf_reg': params['l2_leaf_reg'],
        'random_strength': params['random_strength'],
        'bagging_temperature': params['bagging_temperature'],
        'border_count': params['border_count'],
        'avg_rmse_tr_train': avg_rmse_tr_train,
        'avg_rmse_tr_test': avg_rmse_tr_test,
        'avg_r2_tr_train': avg_r2_tr_train,
        'avg_r2_tr_test': avg_r2_tr_test,
        'avg_rmse_tg_train': avg_rmse_tg_train,
        'avg_rmse_tg_test': avg_rmse_tg_test,
        'avg_r2_tg_train': avg_r2_tg_train,
        'avg_r2_tg_test': avg_r2_tg_test,
        'avg_rmse_ts_train': avg_rmse_ts_train,
        'avg_rmse_ts_test': avg_rmse_ts_test,
        'avg_r2_ts_train': avg_r2_ts_train,
        'avg_r2_ts_test': avg_r2_ts_test,
    })

    # 返回多个目标的平均RMSE和R²
    return avg_rmse_tr_test, avg_rmse_tg_test, avg_rmse_ts_test

# 设置Optuna的研究方向，同时最小化每个目标的RMSE
study = optuna.create_study(directions=["minimize", "minimize", "minimize"])

# 开始优化
study.optimize(objective, n_trials=50)

# 获取最佳 trial
best_trial = study.best_trials[0]

# 打印最佳结果
print("Best hyperparameters:")
print(best_trial.params)

# 保存结果到 CSV 文件
results_df = pd.DataFrame(results)
results_df.to_csv('1optimization_results_with_fixed_sets.csv', index=False)

# 保存最优模型在训练集和测试集的预测结果
train_df = pd.DataFrame(train_predictions)
train_df.to_csv('1best_train_predictions_fixed.csv', index=False)

test_df = pd.DataFrame(test_predictions)
test_df.to_csv('1best_test_predictions_fixed.csv', index=False)

# 打印说明
print(f"Best trial: {best_trial.number}")
print("Best trial predictions saved to 'best_train_predictions_fixed.csv' and 'best_test_predictions_fixed.csv'")
