import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    
    # Target composition columns to predict in the inverse model
    composition_cols = [col for col in feature_cols if col != "Age"]

    # Input features for the inverse model (Performance indicators)
    inverse_input_cols = ["Resistance"]
    if "Age" in df.columns:
        inverse_input_cols = ["Resistance", "Age"]

    X_inverse = df[inverse_input_cols].copy()
    Y_inverse = df[composition_cols].copy()

    print("\nInverse Model Configurations:")
    print("Inputs (Performance):", inverse_input_cols)
    print("Outputs (Composition):", composition_cols)

    # 3. Train-test split (matching the notebook configuration)
    X_inv_train, X_inv_test, Y_inv_train, Y_inv_test = train_test_split(
        X_inverse,
        Y_inverse,
        test_size=0.20,
        random_state=42
    )
    print(f"\nTraining set size: {X_inv_train.shape[0]} samples")
    print(f"Test set size: {X_inv_test.shape[0]} samples")

    # 4. Initialize and fit KNN Regressor pipeline
    print("\nTraining K-Neighbors Regressor Pipeline...")
    inverse_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5))
    ])
    
    inverse_model.fit(X_inv_train, Y_inv_train)

    # 5. Evaluate the model on the test set
    Y_inv_pred = inverse_model.predict(X_inv_test)
    
    # Calculate overall metrics
    inverse_mae = mean_absolute_error(Y_inv_test, Y_inv_pred)
    inverse_rmse = np.sqrt(mean_squared_error(Y_inv_test, Y_inv_pred))

    print("\nEvaluation Metrics on Test Set (Global):")
    print(f"MAE:  {inverse_mae:.3f}")
    print(f"RMSE: {inverse_rmse:.3f}")

    # Print component-wise MAE for finer details
    print("\nComponent-wise Mean Absolute Error:")
    for i, col in enumerate(composition_cols):
        col_mae = mean_absolute_error(Y_inv_test[col], Y_inv_pred[:, i])
        print(f"- {col}: {col_mae:.3f}")

    # 6. Save the model as a pickle file (.pkl)
    output_filename = "concrete_composition_knn_inverse.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(inverse_model, f)
    
    print(f"\nInverse model exported successfully as '{output_filename}' in the same directory.")

if __name__ == "__main__":
    main()
