# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
import numpy as np

# 初始化结果列表
all_results = []
all_feature_subsets = []

# 循环处理20对数据集
for i in range(1, 21):
    train_file = f'train{i}.csv'  # 根据实际路径调整
    test_file = f'test{i}.csv'    # 根据实际路径调整
    
    # 加载训练集和测试集
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    X_train = train_data.iloc[:, 1:]  # 训练集特征
    y_train = train_data.iloc[:, 0]   # 训练集目标变量
    X_test = test_data.iloc[:, 1:]    # 测试集特征
    y_test = test_data.iloc[:, 0]     # 测试集目标变量
    
    # 计算互信息得分
    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    mi_scores = pd.Series(mi_scores, index=X_train.columns)
    
    # 使用所有特征训练初始 CatBoost 模型
    cat_model_full = CatBoostRegressor(random_state=42, verbose=0)
    cat_model_full.fit(X_train, y_train)
    
    # 计算排列重要性得分
    result = permutation_importance(cat_model_full, X_train, y_train,
                                    scoring='neg_mean_squared_error', n_repeats=10, random_state=42, n_jobs=-1)
    pi_scores = pd.Series(result.importances_mean, index=X_train.columns)
    
    # 标准化互信息得分和排列重要性得分
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
    pi_scores_norm = (pi_scores - pi_scores.min()) / (pi_scores.max() - pi_scores.min())
    
    # 计算综合特征重要性得分（可以简单相加，或根据需要调整权重）
    combined_scores = mi_scores_norm + pi_scores_norm  # 或者乘以权重，如 0.5 * mi_scores_norm + 0.5 * pi_scores_norm
    combined_scores = combined_scores.sort_values(ascending=False)
    
    # 特征数量范围
    feature_range = range(20, 4, -1)  # 从20个特征减少到5个特征

    # 存储当前数据集的结果
    results = []
    feature_subsets = []

    for n_features in feature_range:
        print(f"数据集 {i} | 特征数 {n_features} | 开始")
    
        # 选择综合得分最高的前 n_features 个特征
        selected_features = combined_scores.head(n_features).index.tolist()
        feature_subsets.append({
            'Dataset': i,
            'n_features': n_features,
            'Selected_Features': selected_features
        })
    
        # 基于选择的特征重新训练 CatBoost 模型
        cat_model = CatBoostRegressor(random_state=42, verbose=0)
        cat_model.fit(X_train[selected_features], y_train)
    
        # 在测试集上评估模型
        y_pred_test = cat_model.predict(X_test[selected_features])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
    
        # 在训练集上评估模型
        y_pred_train = cat_model.predict(X_train[selected_features])
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
    
        # 十折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_rmse_scores = -cross_val_score(cat_model, X_train[selected_features], y_train,
                                          scoring='neg_root_mean_squared_error', cv=kf, n_jobs=-1)
        cv_r2_scores = cross_val_score(cat_model, X_train[selected_features], y_train, scoring='r2', cv=kf, n_jobs=-1)
        cv_rmse = cv_rmse_scores.mean()
        cv_r2 = cv_r2_scores.mean()
    
        # 保存当前迭代的结果
        results.append({
            'Dataset': i,
            'n_features': n_features,
            'Train_RMSE': train_rmse,
            'Train_R2': train_r2,
            'CV_RMSE': cv_rmse,
            'CV_R2': cv_r2,
            'Test_RMSE': test_rmse,
            'Test_R2': test_r2
        })
        
        print(f"数据集 {i} | 特征数 {n_features} | 结束")
    
    # 将当前数据集的结果添加到总结果列表中
    all_results.extend(results)
    all_feature_subsets.extend(feature_subsets)

# 将所有结果转换为 DataFrame 并打印
all_results_df = pd.DataFrame(all_results)
print(all_results_df)

# 将所有特征子集转换为 DataFrame
all_feature_subsets_df = pd.DataFrame(all_feature_subsets)
print(all_feature_subsets_df)

# 将结果保存为 CSV 文件
all_results_df.to_csv('tg_mi_catboost_results.csv', index=False)
all_feature_subsets_df.to_csv('tg_mi_catboost_feature_subsets.csv', index=False)