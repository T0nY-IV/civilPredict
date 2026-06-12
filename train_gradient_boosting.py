import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    # 1. Load the dataset
    dataset_path = Path("dataset_concrete_formulations_final.csv")
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found in the current directory.")
        return

    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {dataset_path}")
    print(f"Shape: {df.shape}")

    # 2. Select target and feature columns
    target_col = "Resistance"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    print("\nFeatures used:")
    for col in feature_cols:
        print(f"- {col}")
    print(f"Target: {target_col}")

    # 3. Train-test split (matching the notebook configuration)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # 4. Initialize and fit Gradient Boosting model
    print("\nTraining Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate the model on the test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nEvaluation Metrics on Test Set:")
    print(f"MAE:  {mae:.3f} MPa")
    print(f"RMSE: {rmse:.3f} MPa")
    print(f"R²:   {r2:.3f}")

    # 6. Save the model as a pickle file (.pkl)
    output_filename = "concrete_strength_gradient_boosting.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(model, f)
    
    print(f"\nModel exported successfully as '{output_filename}' in the same directory.")

if __name__ == "__main__":
    main()
