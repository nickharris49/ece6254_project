"""
This script explores different regression models to fit the given dataset and evaluates their performance.

### Steps Taken:
1. Linear & Polynomial Regression:
   - Implemented both Linear Regression and Polynomial Regression (degree 2).
   - Evaluated models using Mean Squared Error (MSE) and R² score on a validation set.

2. Outlier Removal:
   - Used Cook's Distance to identify and remove high-influence outliers.

3. Feature Selection & Multicollinearity Reduction:
   - Checked for Variance Inflation Factor (VIF) to detect multicollinearity.
   - Found high VIF values, so Principal Component Analysis (PCA) was applied to reduce dimensions and address multicollinearity.

4. Regression Assumption Checks:
   - Scatter plots showed a non-linear relationship between features and target.
   - Histogram of residuals was imbalanced, indicating non-normality.
   - Residuals vs. Predicted plot showed heteroscedasticity, violating regression assumptions.
   - QQ plot confirmed that residuals were not normally distributed.

5. Feature Transformations Attempted:
   - Applied log transformation, interaction terms, and polynomial expansion to linearize relationships.
   - None of these transformations improved model performance.

### Next Steps:
- Since the dataset does not conform to linear regression assumptions, tree-based models will be explored next:
  - Decision Trees
  - XGBoost (Extreme Gradient Boosting)
  - Potentially other ensemble methods like Random Forest.

"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Step 1: Outlier Detection and Removal
def check_outliers(X_train, y_train):
    model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    outliers = np.where(cooks_d > 4 / len(X_train))[0]

    print(f"Found {len(outliers)} potential outliers.")
    if len(outliers) > 0:
        X_train_cleaned = np.delete(X_train, outliers, axis=0)
        y_train_cleaned = np.delete(y_train, outliers, axis=0)
        return X_train_cleaned, y_train_cleaned
    else:
        return X_train, y_train

# Step 2: Check VIF and Apply PCA if Necessary
def check_vif(X):
    X_const = sm.add_constant(X)
    vif_vals = [variance_inflation_factor(X_const, i) for i in range(X_const.shape[1])]
    for i, vif in enumerate(vif_vals):
        print(f"Feature {i} VIF: {vif:.2f}")
    return vif_vals

def apply_pca(X):
    pca = PCA(n_components=X.shape[1])
    X_pca = pca.fit_transform(X)
    print(f"Explained variance by PCA components: {pca.explained_variance_ratio_}")
    return X_pca

# Step 3: Add Polynomial Features
def add_polynomial_features(X_train, X_val, degree=2):
    print(f"Applying Polynomial Regression (Degree {degree})...")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)  # Transform validation data using same fit
    print(f"Polynomial features added: {X_train_poly.shape[1]} features.")
    return X_train_poly, X_val_poly

# Step 4: Train and Evaluate Models
def train_and_evaluate_models(X_train, X_val, y_train, y_val):
    print("\nTraining Models...")
    start_time = time.time()

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr_val = lr_model.predict(X_val)

    # Polynomial Regression (Degree 2)
    X_train_poly, X_val_poly = add_polynomial_features(X_train, X_val, degree=2)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly_val = poly_model.predict(X_val_poly)

    # Evaluate models
    mse_lr = mean_squared_error(y_val, y_pred_lr_val)
    r2_lr = r2_score(y_val, y_pred_lr_val)

    mse_poly = mean_squared_error(y_val, y_pred_poly_val)
    r2_poly = r2_score(y_val, y_pred_poly_val)

    print("\n--- Model Performance ---")
    print(f"Linear Regression:      MSE={mse_lr:.4f},   R²={r2_lr:.4f}")    
    print(f"Polynomial Regression:  MSE={mse_poly:.4f}, R²={r2_poly:.4f}")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")

# Main Function
def main():
    datadir = "DATASET/"
    
    X_path = datadir + "feature_vector_full.npy"
    y_path = datadir + "y.npy"

    X = np.load(X_path)
    y = np.load(y_path)

    # Split data into train/validation sets (use test set only after final selection)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

    # Convert y_train, y_val, and y_test to 1D arrays
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    y_test = y_test.ravel()

    print(f"Train set shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set shapes: X={X_val.shape}, y={y_val.shape}")
    print(f"Test set shapes: X={X_test.shape}, y={y_test.shape}")

    # Step 1: Remove Outliers from training set
    X_train_cleaned, y_train_cleaned = check_outliers(X_train, y_train)

    # Step 2: Check VIF and Apply PCA if Necessary
    vif_vals_before = check_vif(X_train_cleaned)
    if any(vif > 10 for vif in vif_vals_before):
        print("High VIF detected. Applying PCA...")
        X_transformed = apply_pca(X_train_cleaned)
        vif_vals_after = check_vif(X_transformed)
        print("VIF after PCA:", vif_vals_after)
        X_final = X_transformed
    else:
        X_final = X_train_cleaned

    print(f"Train set shapes after transformation: X={X_final.shape}, y={y_train_cleaned.shape}")

    # Step 3: Train and Evaluate Models using validation set
    train_and_evaluate_models(X_final, X_val, y_train_cleaned, y_val)

if __name__ == '__main__':
    main()
