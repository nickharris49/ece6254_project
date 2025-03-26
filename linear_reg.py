import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

# Step 3: Scatter Plots for Linearity
def scatter_plot_features(X_train, y_train):
    num_features = X_train.shape[1]
    for i in range(num_features):
        plt.figure(figsize=(6, 4))
        plt.scatter(X_train[:, i], y_train, alpha=0.5)
        plt.title(f"Scatter Plot: Feature {i} vs Target")
        plt.xlabel(f"Feature {i}")
        plt.ylabel("Target Variable")
        plt.show()

# Step 4: Add Non-Linearity
def add_non_linear_features(X, method="degree_2"):
    """
    Add non-linear features to the dataset based on the specified method.
    """
    if method == "degree_2":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_transformed = poly.fit_transform(X)
        print("Added degree-2 polynomial features.")
    
    elif method == "degree_3":
        poly = PolynomialFeatures(degree=3, include_bias=False)
        X_transformed = poly.fit_transform(X)
        print("Added degree-3 polynomial features.")
    
    elif method == "interaction":
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        X_transformed = poly.fit_transform(X)
        print("Added interaction terms.")
    
    elif method == "log":
        # Apply logarithmic transformation (handle zero or negative values carefully)
        if np.any(X <= 0):
            print("Log transformation cannot be applied due to zero or negative values in features.")
            return X  # Return original data if log cannot be applied
        X_transformed = np.log(X)
        print("Applied logarithmic transformation.")
    
    elif method == "ihs":
        # Apply Inverse Hyperbolic Sine (IHS) transformation
        X_transformed = np.log(X + np.sqrt(X**2 + 1))
        print("Applied Inverse Hyperbolic Sine (IHS) transformation.")
    
    else:
        raise ValueError("Invalid method specified. Choose from 'degree_2', 'degree_3', 'interaction', 'log', or 'ihs'.")
    
    return X_transformed

# Step 5: Recheck Linear Regression Assumptions
def check_linear_regression_assumptions(X_train, y_train):
    model = LinearRegression() 
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    residuals = y_train - y_pred

    # Histogram of residuals (Normality Check)
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, edgecolor='k')
    plt.title("Histogram of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.show()

    # Residuals vs Predicted (Homoscedasticity Check)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals vs Predicted")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.show()

    # QQ Plot (Normality Check)
    sm.qqplot(residuals, line='45', fit=True)
    plt.title("QQ Plot of Residuals")
    plt.show()

def main():
    datadir = "DATASET/"
    
    X_path = datadir + "feature_vector_full.npy"
    y_path = datadir + "y.npy"

    X = np.load(X_path)
    y = np.load(y_path)

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print shapes of train and test sets
    print(f"Train set shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shapes: X={X_test.shape}, y={y_test.shape}")

    # Step 1: Remove Outliers
    X_train_cleaned, y_train_cleaned = check_outliers(X_train, y_train)
    print(f"Train set shapes: X={X_train_cleaned.shape}, y={y_train_cleaned.shape}")

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
    
    print(f"Train set shapes: X={X_final.shape}, y={y_train.shape}")

    # Step 3: Scatter Plots for Linearity
    # scatter_plot_features(X_final, y_train_cleaned)

    # Step 4: Add Non-Linearity
    X_poly = add_non_linear_features(X_final, method="ihs")

    # Step 5: Recheck Linear Regression Assumptions
    scatter_plot_features(X_poly, y_train_cleaned)
    # check_linear_regression_assumptions(X_poly, y_train_cleaned)

if __name__ == '__main__':
    main()
