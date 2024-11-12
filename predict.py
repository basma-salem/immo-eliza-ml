import pandas as pd
import joblib
from Preprocessing import DataProcessor


def Read_model_data(model_path): #Read model data From 
    return joblib.load(model_path)

def preprocess_new_data(data_path): #preprocess the data
    preprocessor = DataProcessor(data_path)
    preprocessed_data = preprocessor.preprocess()
    return preprocessed_data

def main():
    '''Read the model and process the data the predict the prices and save it'''
    model = Read_model_data('data\\xgboost_model.joblib')
    #Preprocess the data
    new_data = preprocess_new_data('data\properties.csv').drop(
        'price', axis=1, errors='ignore')
    
    # Make predictions on the new data
    predictions = model.predict(new_data)
    
    #save the pridictions
    pd.DataFrame(predictions, columns=['Predicted Price']).to_csv(
    'data\predictions.csv', index=False)
    print(f"Predictions saved :)")

if __name__ == "__main__":
    main()