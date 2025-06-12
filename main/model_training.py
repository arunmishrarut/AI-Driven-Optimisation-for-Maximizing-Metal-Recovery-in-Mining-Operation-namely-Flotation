import numpy as np
from cuml.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def train_models(X_train_s, y_train_s, X_train_a, y_train_a):
    model_starch = RandomForestRegressor({
        'n_estimators': 100,
        'max_depth': 16,
        'n_streams': 1,
        'random_state': 0
    })
    model_starch.fit(X_train_s, y_train_s)

    model_amina = RandomForestRegressor({
        'n_estimators': 100,
        'max_depth': 16,
        'n_streams': 1,
        'random_state': 0
    })
    model_amina.fit(X_train_a, y_train_a)

    return model_starch, model_amina

def evaluate_models(model_starch, model_amina, X_test_s, y_test_s, X_test_a, y_test_a):
    pred_s = model_starch.predict(X_test_s)
    pred_a = model_amina.predict(X_test_a)

    print("R² Score - Starch Flow:", r2_score(y_test_s, pred_s))
    print("R² Score - Amina Flow:", r2_score(y_test_a, pred_a))

def make_sample_predictions(model_starch, model_amina, X_test_s, y_test_s, X_test_a, y_test_a):
    sample_starch = np.array(X_test_s.iloc[0], dtype=np.float32)
    sample_amina = np.array(X_test_a.iloc[0], dtype=np.float32)

    starch_pred = model_starch.predict(sample_starch)
    amina_pred = model_amina.predict(sample_amina)

    print("Sample Starch Flow prediction:", starch_pred)
    print("Actual:", y_test_s.iloc[0])
    print("Sample Amina Flow prediction:", amina_pred)
    print("Actual:", y_test_a.iloc[0])
