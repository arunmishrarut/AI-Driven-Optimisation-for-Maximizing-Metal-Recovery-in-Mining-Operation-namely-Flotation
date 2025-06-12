from data_loader import load_data
from preprocessing import prepare_features_and_targets
from model_training import train_models, evaluate_models, make_sample_predictions
from interface import launch_interface


def main():
    # Load data
    df = load_data("train_data.csv")

    # Prepare features and targets
    (
        features_starch, features_amina,
        X_train_s, X_test_s, y_train_s, y_test_s,
        X_train_a, X_test_a, y_train_a, y_test_a
    ) = prepare_features_and_targets(df)

    # Train models
    model_starch, model_amina = train_models(
        X_train_s, y_train_s,
        X_train_a, y_train_a
    )

    # Evaluate models
    evaluate_models(
        model_starch, model_amina,
        X_test_s, y_test_s,
        X_test_a, y_test_a
    )

    # Sample predictions
    make_sample_predictions(
        model_starch, model_amina,
        X_test_s, y_test_s,
        X_test_a, y_test_a
    )

    # Launch UI
    launch_interface(model_starch, model_amina, features_starch, features_amina)


if __name__ == "__main__":
    main()
