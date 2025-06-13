from data_loader import load_data
from preprocessing import prepare_features_and_targets
from model_training import train_model, evaluate_model
from interface import launch_interface


def main():
    df = load_data("train_data.csv")
    features_amina, X_train, X_test, y_train, y_test = prepare_features_and_targets(df)
    model_amina = train_model(X_train, y_train)
    evaluate_model(model_amina, X_test, y_test)
    launch_interface(model_amina, features_amina)


if __name__ == "__main__":
    main()
