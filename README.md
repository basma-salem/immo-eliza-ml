# Immo Price Prediction 
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Numpy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🏢 Description
Welcome to my first Machine Learning project! I used a dataset of houses that scraped from the Immoweb.be and in this repo 
I tried some regressorssuch Linear and DecessionTree but XGBOOST was the best to my dataset to predict the price of a house based on its features.


## 📦 Repo structure
```
immo-eliza-ml-ML/
│
├── data/
│   └── properties.csv                  --- data: training dataset (.csv)
    └──xgboost_model.joblib             --- trained models saved in .joblib format to use for predictions
├── preprocessing.ipynp                 --- data preprocessing  and figiure out  the models to deciede the best
├── preprocessing.py                    --- data preprocessing Class to clean, impute & encode data
├── training.py                         --- the base training Class, that all models inherit from
├── predict.py                          --- runs predictions on new dataset(s) and saves output in .csv format
├── predictions.csv                     --- predictions data (.csv)
├── .gitignore
├── requirements.txt
└── README.md
```
## 🔧 Installation

Follow these steps to set up your project environment:

- **Clone the repository**
  `git clone https://github.com/basma-salem/immo-eliza-ml.git`
- **Navigate to the project directory**
  `cd immo-eliza-ml`
- **Install the required dependencies**
  `pip install -r requirements.txt`t


## 🚀 To train a model
I have iteratively tested 1. various preprocessing regressors and 2. various algorithms to train and evaluate multiple ML models. I found that the  XGBoost Regressor model performed the best for my dataset.
Here is an example of the evaluation results for the test set, with three different algorithms:

```

| Model           | MAE         |   RMSE      |  Training score |    R2      |

| LinearRegressor | 172,919.83  | 337,468.44  |      0.33        |  0.39    |
| DecessionTree   | 112,838.57 | 247,415.78   |      0.82        |  0.67    |
| XGBoost         | 90,808.14  | 186,834.96   |      0.95        |  0.81    |

```
### Planned Upgrades
- **Data Pipeline Enhancement**: Improve the automation of data preprocessing and feature selection.

## ⏱️ Timeline
This project was done in 5 days including studying the theory and implementing the code.

## 📌 Personal Situation
This project was done as part of my AI training program at BeCode.


### Connect with me!
[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/basma-salem-ba45a1113)