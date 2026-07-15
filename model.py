# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, HuberRegressor)
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
# from sklearn.neural_network import MLPRegressor
# from sklearn.neighbors import KNeighborsRegressor
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import pickle
# import os

# # Ensure 'models' folder exists
# os.makedirs('models', exist_ok=True)

# # Load dataset
# data = pd.read_csv(r'clean_dataset.csv')

# # --- FIX 1: Remove 'Unnamed: 0' if it exists in the raw data ---
# if 'Unnamed: 0' in data.columns:
#     data = data.drop(['Unnamed: 0'], axis=1)
# # ---------------------------------------------------------------

# # Preprocessing
# X = data.drop(['quality'], axis=1) 
# y = data['quality']

# X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Define models
# models = {
#     'LinearRegression': LinearRegression(),
#     'RobustRegression': HuberRegressor(),
#     'RidgeRegression': Ridge(),
#     'LassoRegression': Lasso(),
#     'ElasticNet': ElasticNet(),
#     'PolynomialRegression': Pipeline([
#         ('poly', PolynomialFeatures(degree=2)),
#         ('linear', LinearRegression())
#     ]),
#     'SGDRegressor': SGDRegressor(),
#     'ANN': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000),
#     'RandomForest': RandomForestRegressor(),
#     'SVM': SVR(),
#     'LGBM': lgb.LGBMRegressor(),
#     'XGBoost': xgb.XGBRegressor(),
#     'KNN': KNeighborsRegressor()
# }

# # Train and evaluate models
# results = []

# for name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     mae = mean_absolute_error(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    
#     results.append({
#         'Model': name,
#         'MAE': mae,
#         'MSE': mse,
#         'R2': r2
#     })
    
#     # Save each model inside 'models' folder
#     with open(f'models/{name}.pkl', 'wb') as f:
#         pickle.dump(model, f)

# # Convert results to DataFrame and save to CSV inside 'models' folder
# results_df = pd.DataFrame(results)

# # --- FIX 2: Ensure the results CSV doesn't generate its own 'Unnamed: 0' next time ---
# results_df.to_csv('models/model_evaluation_results.csv', index=False)
# # -------------------------------------------------------------------------------------

# print("✅ Models have been trained and saved in the 'models' folder. Evaluation results saved to models/model_evaluation_results.csv.")