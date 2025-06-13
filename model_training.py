import numpy as np
from cuml.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

def train_model(X_train, y_train):
    model = RandomForestRegressor({
        'n_estimators': 100,
        'max_depth': 16,
        'n_streams': 1,
        'random_state': 0
    })
    model.fit(X_train, y_train)
    joblib.dump(model, "model_amina.joblib")
    print("✅ Model saved as 'model_amina.joblib'")
    return model

def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    print("R² Score - Amina Flow:", r2_score(y_test, pred))


