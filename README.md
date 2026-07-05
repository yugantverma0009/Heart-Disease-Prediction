# Heart Disease Prediction

This project predicts the likelihood of heart disease using machine learning. The model is trained on the Heart Disease dataset and can be tested using a Streamlit web app.

The project includes model training, evaluation, and a simple Streamlit interface for predictions.

## Features

- Train multiple ML models
- Compare model performance
- Save the best model
- Predict using a Streamlit web app

## Dataset

Place the dataset inside:

```text
data/heart.csv
```

## Installation

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

## Run

```bash
streamlit run app.py
```

## Models Used

- Logistic Regression
- Random Forest
- Decision Tree
- KNN
- Gradient Boosting
- AdaBoost

## Note

This is a college project and should not be used for real medical decisions.
