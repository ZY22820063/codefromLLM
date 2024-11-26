import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# 加载数据
data = pd.read_csv('3T.csv')
X = data.iloc[:, 3:].values  # 假设前三列是目标，后面是特征
y_tr = data['tr'].values
y_tg = data['tg'].values
y_ts = data['ts'].values

# 设置10折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 用于存储结果的列表
results = []

# 用于存储每个 trial 的训练集和测试集预测结果
train_predictions = []
test_predictions = []

# 定义目标函数，用于贝叶斯优化
def objective(trial):
    # 选择一组超参数
    params = {
        'iterations': trial.suggest_int('iterations', 10, 2000),
        'depth': trial.suggest_int('depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 100),
        'random_strength': trial.suggest_float('random_strength', 1e-5, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
        'border_count': trial.suggest_int('border_count', 32, 512),
        'verbose': 0
    }

    # 初始化RMSE和R2存储
    avg_rmse_tr_train = 0
    avg_rmse_tr_test = 0
    avg_r2_tr_train = 0
    avg_r2_tr_test = 0

    avg_rmse_tg_train = 0
    avg_rmse_tg_test = 0
    avg_r2_tg_train = 0
    avg_r2_tg_test = 0

    avg_rmse_ts_train = 0
    avg_rmse_ts_test = 0
    avg_r2_ts_train = 0
    avg_r2_ts_test = 0

    fold_train_predictions = []
    fold_test_predictions = []

    # 进行10折交叉验证
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_tr, y_test_tr = y_tr[train_idx], y_tr[test_idx]
        y_train_tg, y_test_tg = y_tg[train_idx], y_tg[test_idx]
        y_train_ts, y_test_ts = y_ts[train_idx], y_ts[test_idx]

        # 训练模型 tr
        model_tr = CatBoostRegressor(**params, random_state=42)
        model_tr.fit(X_train, y_train_tr)
        y_pred_tr_train = model_tr.predict(X_train)
        y_pred_tr_test = model_tr.predict(X_test)
        avg_rmse_tr_train += np.sqrt(mean_squared_error(y_train_tr, y_pred_tr_train))
        avg_rmse_tr_test += np.sqrt(mean_squared_error(y_test_tr, y_pred_tr_test))
        avg_r2_tr_train += r2_score(y_train_tr, y_pred_tr_train)
        avg_r2_tr_test += r2_score(y_test_tr, y_pred_tr_test)

        # 训练模型 tg
        model_tg = CatBoostRegressor(**params, random_state=42)
        model_tg.fit(X_train, y_train_tg)
        y_pred_tg_train = model_tg.predict(X_train)
        y_pred_tg_test = model_tg.predict(X_test)
        avg_rmse_tg_train += np.sqrt(mean_squared_error(y_train_tg, y_pred_tg_train))
        avg_rmse_tg_test += np.sqrt(mean_squared_error(y_test_tg, y_pred_tg_test))
        avg_r2_tg_train += r2_score(y_train_tg, y_pred_tg_train)
        avg_r2_tg_test += r2_score(y_test_tg, y_pred_tg_test)

        # 训练模型 ts
        model_ts = CatBoostRegressor(**params, random_state=42)
        model_ts.fit(X_train, y_train_ts)
        y_pred_ts_train = model_ts.predict(X_train)
        y_pred_ts_test = model_ts.predict(X_test)
        avg_rmse_ts_train += np.sqrt(mean_squared_error(y_train_ts, y_pred_ts_train))
        avg_rmse_ts_test += np.sqrt(mean_squared_error(y_test_ts, y_pred_ts_test))
        avg_r2_ts_train += r2_score(y_train_ts, y_pred_ts_train)
        avg_r2_ts_test += r2_score(y_test_ts, y_pred_ts_test)

        # 保存训练集和测试集的真实值和预测值
        fold_train_predictions.append({
            'y_true_tr': y_train_tr, 'y_pred_tr': y_pred_tr_train,
            'y_true_tg': y_train_tg, 'y_pred_tg': y_pred_tg_train,
            'y_true_ts': y_train_ts, 'y_pred_ts': y_pred_ts_train
        })
        fold_test_predictions.append({
            'y_true_tr': y_test_tr, 'y_pred_tr': y_pred_tr_test,
            'y_true_tg': y_test_tg, 'y_pred_tg': y_pred_tg_test,
            'y_true_ts': y_test_ts, 'y_pred_ts': y_pred_ts_test
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

    # 返回训练集和测试集的RMSE和R2作为目标
    return avg_rmse_tr_train, avg_rmse_tg_train, avg_rmse_ts_train, avg_rmse_tr_test, avg_rmse_tg_test, avg_rmse_ts_test

# 设置Optuna的研究方向，同时最小化训练集和测试集的RMSE
study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize", "minimize", "minimize"])

# 开始优化
study.optimize(objective, n_trials=50)

# 获取最佳 Trial
best_trial = study.best_trials[0]
best_trial_number = best_trial.number

# 打印最佳结果
print("Best hyperparameters:")
print(best_trial.params)

# 打印所有结果，并保存到 CSV 文件
results_df = pd.DataFrame(results)

# 保存到 CSV 文件
results_df.to_csv('optimization_results_with_avgmore.csv', index=False)

# **以下是修改的部分**

# 获取最佳 Trial 的预测结果
best_train_predictions = train_predictions[best_trial_number]
best_test_predictions = test_predictions[best_trial_number]

# 初始化字典用于存储所有折叠的真实值和预测值
# 测试集
test_y_true_tr_all_folds = []
test_y_pred_tr_all_folds = []
test_y_true_tg_all_folds = []
test_y_pred_tg_all_folds = []
test_y_true_ts_all_folds = []
test_y_pred_ts_all_folds = []

# 训练集
train_y_true_tr_all_folds = []
train_y_pred_tr_all_folds = []
train_y_true_tg_all_folds = []
train_y_pred_tg_all_folds = []
train_y_true_ts_all_folds = []
train_y_pred_ts_all_folds = []

# 处理每个折叠的数据
def adjust_fold_length(y_true_all_folds, y_pred_all_folds):
    """
    调整每个折叠的长度，使所有折叠的数据长度一致
    """
    min_length = min([len(fold) for fold in y_true_all_folds])  # 找到最小长度
    y_true_adjusted = [fold[:min_length] for fold in y_true_all_folds]  # 截取到相同长度
    y_pred_adjusted = [fold[:min_length] for fold in y_pred_all_folds]  # 截取到相同长度
    return y_true_adjusted, y_pred_adjusted

# 遍历最佳 trial 的每个折叠，收集真实值和预测值
for fold in best_train_predictions:
    train_y_true_tr_all_folds.append(fold['y_true_tr'])
    train_y_pred_tr_all_folds.append(fold['y_pred_tr'])
    train_y_true_tg_all_folds.append(fold['y_true_tg'])
    train_y_pred_tg_all_folds.append(fold['y_pred_tg'])
    train_y_true_ts_all_folds.append(fold['y_true_ts'])
    train_y_pred_ts_all_folds.append(fold['y_pred_ts'])

for fold in best_test_predictions:
    test_y_true_tr_all_folds.append(fold['y_true_tr'])
    test_y_pred_tr_all_folds.append(fold['y_pred_tr'])
    test_y_true_tg_all_folds.append(fold['y_true_tg'])
    test_y_pred_tg_all_folds.append(fold['y_pred_tg'])
    test_y_true_ts_all_folds.append(fold['y_true_ts'])
    test_y_pred_ts_all_folds.append(fold['y_pred_ts'])

# 调整训练集和测试集每个折叠的长度一致
train_y_true_tr_all_folds, train_y_pred_tr_all_folds = adjust_fold_length(train_y_true_tr_all_folds, train_y_pred_tr_all_folds)
train_y_true_tg_all_folds, train_y_pred_tg_all_folds = adjust_fold_length(train_y_true_tg_all_folds, train_y_pred_tg_all_folds)
train_y_true_ts_all_folds, train_y_pred_ts_all_folds = adjust_fold_length(train_y_true_ts_all_folds, train_y_pred_ts_all_folds)

test_y_true_tr_all_folds, test_y_pred_tr_all_folds = adjust_fold_length(test_y_true_tr_all_folds, test_y_pred_tr_all_folds)
test_y_true_tg_all_folds, test_y_pred_tg_all_folds = adjust_fold_length(test_y_true_tg_all_folds, test_y_pred_tg_all_folds)
test_y_true_ts_all_folds, test_y_pred_ts_all_folds = adjust_fold_length(test_y_true_ts_all_folds, test_y_pred_ts_all_folds)

# 将每个样本在10折交叉验证中的预测值取平均，真实值保持不变
# 训练集
train_avg_y_pred_tr = np.mean(np.array(train_y_pred_tr_all_folds), axis=0)
train_avg_y_pred_tg = np.mean(np.array(train_y_pred_tg_all_folds), axis=0)
train_avg_y_pred_ts = np.mean(np.array(train_y_pred_ts_all_folds), axis=0)

train_y_true_tr = np.array(train_y_true_tr_all_folds[0])  # 真实值保持不变
train_y_true_tg = np.array(train_y_true_tg_all_folds[0])
train_y_true_ts = np.array(train_y_true_ts_all_folds[0])

# 测试集
test_avg_y_pred_tr = np.mean(np.array(test_y_pred_tr_all_folds), axis=0)
test_avg_y_pred_tg = np.mean(np.array(test_y_pred_tg_all_folds), axis=0)
test_avg_y_pred_ts = np.mean(np.array(test_y_pred_ts_all_folds), axis=0)

test_y_true_tr = np.array(test_y_true_tr_all_folds[0])  # 真实值保持不变
test_y_true_tg = np.array(test_y_true_tg_all_folds[0])
test_y_true_ts = np.array(test_y_true_ts_all_folds[0])

# 创建 DataFrame 保存训练集和测试集每个样本的真实值和预测值均值
train_avg_df = pd.DataFrame({
    'avg_y_true_tr': train_y_true_tr,
    'avg_y_pred_tr': train_avg_y_pred_tr,
    'avg_y_true_tg': train_y_true_tg,
    'avg_y_pred_tg': train_avg_y_pred_tg,
    'avg_y_true_ts': train_y_true_ts,
    'avg_y_pred_ts': train_avg_y_pred_ts
})

test_avg_df = pd.DataFrame({
    'avg_y_true_tr': test_y_true_tr,
    'avg_y_pred_tr': test_avg_y_pred_tr,
    'avg_y_true_tg': test_y_true_tg,
    'avg_y_pred_tg': test_avg_y_pred_tg,
    'avg_y_true_ts': test_y_true_ts,
    'avg_y_pred_ts': test_avg_y_pred_ts
})

# 保存训练集和测试集每个样本的均值到CSV文件
#train_avg_df.to_csv('best_train_avg_predictions_per_sample1.csv', index=False)
#test_avg_df.to_csv('best_test_avg_predictions_per_sample1.csv', index=False)

# 打印说明
print("训练集和测试集每个样本的真实值和预测值的均值已保存为 'best_train_avg_predictions_per_sample.csv' 和 'best_test_avg_predictions_per_sample.csv'")