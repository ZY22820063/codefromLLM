# codefromLLM
Code from ChatGPT4. A full process for predicting the properties of polyimides with high temperature, strength and permeability, including data collection, data preprocessing, feature screening, modelling and high-throughput prediction.
1. Descriptor generation
MFF.py--input list-A.csv, the first column is number, the second column is smiles
2. Descriptor integration
connect.py--input a-connect.csv, the index and molar ratio of dianhydride and diamine; input a-mff-connect.csv, the first column is index, and the rest are mff
3.preprocessing
pre1.py and preprocessing.py
Filled_Thickness.py
4. Feature engineering
select-mi-catboost.py
5. Model evaluation
catboost-model-tg; catboost-model-ts; catboost-model-tr
6. Parameter optimization
catboost-parameter-tg-peomote.py
7. 3T model
select-rfecatboost-para-3T.py, optuna-catb-parameter.py and catboost-3T.py
8. High-throughput verification
3T-predict.py
