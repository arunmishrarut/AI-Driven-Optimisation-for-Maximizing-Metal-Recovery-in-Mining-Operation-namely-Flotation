import statsmodels.api as sm
from sklearn.model_selection import train_test_split

def prepare_features_and_targets(df):
    # Drop non-numeric or non-useful columns
    features = df.drop(columns=["date", "Starch Flow", "Amina Flow"])
    target_starch = df["Starch Flow"]
    target_amina = df["Amina Flow"]

    # Add constant for regression p-values
    X_const = sm.add_constant(features)

    # Fit models for p-values (not used further, just for printing)
    model_starch = sm.OLS(target_starch, X_const).fit()
    model_amina = sm.OLS(target_amina, X_const).fit()

    print("\nStarch Flow p-values:\n", model_starch.pvalues.sort_values())
    print("\nAmina Flow p-values:\n", model_amina.pvalues.sort_values())

    # Manual selection of features
    features_starch = df[[
        "% Iron Feed", "Ore Pulp Flow", "Ore Pulp Density", "Ore Pulp pH",
        "Flotation Column 01 Level", "Flotation Column 03 Air Flow",
        "Flotation Column 03 Level", "Flotation Column 02 Level", "Flotation Column 04 Level"
    ]]

    features_amina = df[[
        "Ore Pulp Flow", "Ore Pulp Density", "Flotation Column 01 Level",
        "Flotation Column 03 Level", "% Silica Concentrate", "Flotation Column 06 Level"
    ]]

    # Targets
    y_starch = df["Starch Flow"]
    y_amina = df["Amina Flow"]

    # Split data
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(features_starch, y_starch, test_size=0.15, random_state=0)
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(features_amina, y_amina, test_size=0.15, random_state=0)

    return features_starch, features_amina, X_train_s, X_test_s, y_train_s, y_test_s, X_train_a, X_test_a, y_train_a, y_test_a
