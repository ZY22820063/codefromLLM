import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE

# 用于存储所有 trial 的结果
results = []

# 定义目标函数
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

    # 初始化存储每个 trial 中所有数据集和 RFE 步骤的结果
    trial_results = []

    # 循环处理 20 对数据集
    for i in range(1, 21):
        train_file = f'train{i}.csv'
        test_file = f'test{i}.csv'

        # 加载训练集和测试集
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

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

        # 特征从 22 减少到 4
        for n_features_to_select in range(X_train.shape[1], 3, -1):
            cv_rmse_tr_train, cv_rmse_tr_test = [], []
            cv_rmse_tg_train, cv_rmse_tg_test = [], []
            cv_rmse_ts_train, cv_rmse_ts_test = [], []
            cv_r2_tr_train, cv_r2_tr_test = [], []
            cv_r2_tg_train, cv_r2_tg_test = [], []
            cv_r2_ts_train, cv_r2_ts_test = [], []

            for train_idx, test_idx in kf.split(X_train):
                X_cv_train, X_cv_test = X_train[train_idx], X_train[test_idx]
                y_cv_train_tr, y_cv_test_tr = y_train_tr[train_idx], y_train_tr[test_idx]
                y_cv_train_tg, y_cv_test_tg = y_train_tg[train_idx], y_train_tg[test_idx]
                y_cv_train_ts, y_cv_test_ts = y_train_ts[train_idx], y_train_ts[test_idx]

                # 特征选择
                selector = RFE(CatBoostRegressor(**params, random_state=42), n_features_to_select=n_features_to_select, step=1)
                selector.fit(X_cv_train, y_cv_train_tr)  # 基于 tr 进行特征选择

                # 选择的特征
                selected_features = selector.support_
                X_cv_train_selected = X_cv_train[:, selected_features]
                X_cv_test_selected = X_cv_test[:, selected_features]

                # 训练和预测 tr
                model_tr = CatBoostRegressor(**params, random_state=42)
                model_tr.fit(X_cv_train_selected, y_cv_train_tr)
                y_pred_tr_train = model_tr.predict(X_cv_train_selected)
                y_pred_tr_test = model_tr.predict(X_cv_test_selected)

                # 计算 RMSE 和 R²
                rmse_tr_train = np.sqrt(mean_squared_error(y_cv_train_tr, y_pred_tr_train))
                rmse_tr_test = np.sqrt(mean_squared_error(y_cv_test_tr, y_pred_tr_test))
                r2_tr_train = r2_score(y_cv_train_tr, y_pred_tr_train)
                r2_tr_test = r2_score(y_cv_test_tr, y_pred_tr_test)

                cv_rmse_tr_train.append(rmse_tr_train)
                cv_rmse_tr_test.append(rmse_tr_test)
                cv_r2_tr_train.append(r2_tr_train)
                cv_r2_tr_test.append(r2_tr_test)

                # 训练和预测 tg
                model_tg = CatBoostRegressor(**params, random_state=42)
                model_tg.fit(X_cv_train_selected, y_cv_train_tg)
                y_pred_tg_train = model_tg.predict(X_cv_train_selected)
                y_pred_tg_test = model_tg.predict(X_cv_test_selected)

                rmse_tg_train = np.sqrt(mean_squared_error(y_cv_train_tg, y_pred_tg_train))
                rmse_tg_test = np.sqrt(mean_squared_error(y_cv_test_tg, y_pred_tg_test))
                r2_tg_train = r2_score(y_cv_train_tg, y_pred_tg_train)
                r2_tg_test = r2_score(y_cv_test_tg, y_pred_tg_test)

                cv_rmse_tg_train.append(rmse_tg_train)
                cv_rmse_tg_test.append(rmse_tg_test)
                cv_r2_tg_train.append(r2_tg_train)
                cv_r2_tg_test.append(r2_tg_test)

                # 训练和预测 ts
                model_ts = CatBoostRegressor(**params, random_state=42)
                model_ts.fit(X_cv_train_selected, y_cv_train_ts)
                y_pred_ts_train = model_ts.predict(X_cv_train_selected)
                y_pred_ts_test = model_ts.predict(X_cv_test_selected)

                rmse_ts_train = np.sqrt(mean_squared_error(y_cv_train_ts, y_pred_ts_train))
                rmse_ts_test = np.sqrt(mean_squared_error(y_cv_test_ts, y_pred_ts_test))
                r2_ts_train = r2_score(y_cv_train_ts, y_pred_ts_train)
                r2_ts_test = r2_score(y_cv_test_ts, y_pred_ts_test)

                cv_rmse_ts_train.append(rmse_ts_train)
                cv_rmse_ts_test.append(rmse_ts_test)
                cv_r2_ts_train.append(r2_ts_train)
                cv_r2_ts_test.append(r2_ts_test)

            # 保存每一步的结果
            trial_results.append({
                'data_set': i,
                'n_features': n_features_to_select,
                'selected_features': selected_features.tolist(),
                'params': params,
                'avg_rmse_tr_train': np.mean(cv_rmse_tr_train),
                'avg_rmse_tr_test': np.mean(cv_rmse_tr_test),
                'avg_rmse_tg_train': np.mean(cv_rmse_tg_train),
                'avg_rmse_tg_test': np.mean(cv_rmse_tg_test),
                'avg_rmse_ts_train': np.mean(cv_rmse_ts_train),
                'avg_rmse_ts_test': np.mean(cv_rmse_ts_test),
                'avg_r2_tr_train': np.mean(cv_r2_tr_train),
                'avg_r2_tr_test': np.mean(cv_r2_tr_test),
                'avg_r2_tg_train': np.mean(cv_r2_tg_train),
                'avg_r2_tg_test': np.mean(cv_r2_tg_test),
                'avg_r2_ts_train': np.mean(cv_r2_ts_train),
                'avg_r2_ts_test': np.mean(cv_r2_ts_test)
            })

    # 保存所有 trial 的结果
    results.append({
        'trial': trial.number,
        'params': params,
        'trial_results': trial_results
    })

    # 返回平均 tr 训练集 RMSE（可根据需要更改为其他目标指标）
    return np.mean([res['avg_rmse_tr_train'] for res in trial_results])

# 设置 Optuna 的研究方向
study = optuna.create_study(direction="minimize")

# 开始优化
study.optimize(objective, n_trials=50)

# 获取最佳 Trial
best_trial = study.best_trial

# 打印最佳结果
print("Best hyperparameters:")
print(best_trial.params)

# 保存所有 trial 的结果到 CSV 文件
results_df = pd.DataFrame(results)
results_df.to_csv('optimization_results_with_rfe_cv.csv', index=False)
