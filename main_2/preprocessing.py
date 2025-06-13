import statsmodels.api as sm
from sklearn.model_selection import train_test_split

def prepare_features_and_targets(df):
    # Drop non-numeric or non-useful columns
    features = df.drop(columns=["date", "Amina Flow"])
    target_amina = df["Amina Flow"]


    y_amina = df["Amina Flow"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, y_amina, test_size=0.15, random_state=0
    )

    return features, X_train, X_test, y_train, y_test
