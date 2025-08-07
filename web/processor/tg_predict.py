import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from web.dto.SampleData import SampleData

features_mapping = {
    "thickness": "Thickness",
    "time": "Total Imidization Time",
    "temperature": "Maximum Imidization Temperature",
    "wavelength": "Wavelength (nm)"
}

# 选定的特征
selected_features = ['Thickness', 'Total Imidization Time', 'g2434951923', 'a864674487',
                     'Maximum Imidization Temperature', 'g951226070', 'g4290259127',
                     'g3217380708', 'g2968968094', 'a3217380708']

def tg_predict(sample: SampleData):
    # 加载数据集
    train_data = pd.read_csv('./train12-tg.csv')

    # 合并字典，确保值是列表
    input_args = {
        **sample.summed_fingerprints,
        "thickness": sample.thickness,
        "time": sample.time,
        "temperature": sample.temperature,
    }

    # 确保所有值是列表形式
    fingerprints_dict = {
        k: [v] if not isinstance(v, list) else v
        for k, v in input_args.items()
    }

    # 构造 DataFrame
    test_data = pd.DataFrame(fingerprints_dict)
    test_data = test_data.rename(columns=features_mapping)
    # 确保所有 selected_features 存在，缺失列补充为 0
    test_data = test_data.reindex(columns=selected_features, fill_value=0)

    # 分离特征和目标
    X_train = train_data[selected_features]
    y_train = train_data.iloc[:, 0]
    X_test = test_data[selected_features]

    # 定义模型参数
    params = {
        'learning_rate': 0.3,
        'depth': 4,
        'subsample': 0.6,
        'iterations': 100,
        'l2_leaf_reg': 5,
        "save_snapshot": False,
        "logging_level": "Silent",
        "allow_writing_files": False
    }

    # 初始化CatBoost回归器
    catboost = CatBoostRegressor(**params, random_state=42)

    # 使用五折交叉验证进行模型评估
    kf = KFold(n_splits=10)
    train_r2_scores = []
    test_r2_scores = []
    train_rmse_scores = []
    test_rmse_scores = []

    for train_index, test_index in kf.split(X_train):
        X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        catboost.fit(X_train_cv, y_train_cv)

        # 计算训练集和验证集的R2和RMSE
        y_pred_train_cv = catboost.predict(X_train_cv)
        y_pred_test_cv = catboost.predict(X_test_cv)
        train_rmse_scores.append(np.sqrt(mean_squared_error(y_train_cv, y_pred_train_cv)))
        test_rmse_scores.append(np.sqrt(mean_squared_error(y_test_cv, y_pred_test_cv)))
        train_r2_scores.append(r2_score(y_train_cv, y_pred_train_cv))
        test_r2_scores.append(r2_score(y_test_cv, y_pred_test_cv))

    # 使用所有训练数据重新训练模型
    catboost.fit(X_train, y_train)

    y_pred_test = catboost.predict(X_test)

    return y_pred_test[0]
