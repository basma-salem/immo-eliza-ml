import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , GridSearchCV
from xgboost import XGBRegressor
import joblib
from Preprocessing import DataProcessor

def get_CV_regressior(X_train, y_train): #Get best params for XGBRegressor
    params = {
        'max_depth': [3, 5, 10],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }
    model = XGBRegressor(
    objective='reg:squarederror',  # Loss function for regression
    random_state=42                # Seed for reproducibility
    )
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

def standrization(X_train, X_test): # Standardizing each feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    return X_train, X_test

def train_model(df): #Train and fit the model 
    X = df.drop(columns=['price'])
    y = df['price']
    # Splitting data into train and test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=41, test_size=0.2)
    #Standardizing
    X_train, X_test = standrization(X_train, X_test)
    
    #Get best Parametrs for my model XGBRegressor
    best_prames_for_model = get_CV_regressior(X_train, y_train)
    model = XGBRegressor(**best_prames_for_model, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    
    #Calculate errors
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Training score: {train_score:.2f}")
    print(f"Testing Score: {test_score:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    #save model in joblib
    joblib.dump(model, 'data\\xgboost_model.joblib')
    
    
def main():
# Initialize the DataPreprocessor
    preprocessor = DataProcessor('data\properties.csv')
    # Preprocess the data
    preprocessed_data = preprocessor.preprocess()
    # Use the preprocessed_data to train model
    train_model(preprocessed_data)
    
if __name__ == "__main__":
    main()