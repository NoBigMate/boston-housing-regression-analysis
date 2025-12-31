import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset from the original source
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(data, columns=feature_names)
df['MEDV'] = target

print(df.head())

# 1. Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Boston Housing Features")
plt.show()

# Selecting features: We look for features highly correlated with target (MEDV)
# but not highly correlated with each other. 
# RM (0.7) and LSTAT (-0.74) are strong predictors.

# 2. Multicollinearity Check using Variance Inflation Factor (VIF)
X_vif = df.drop(['MEDV'], axis=1)
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
print("\nVIF Data (VIF > 10 indicates high multicollinearity):")
print(vif_data.sort_values(by="VIF", ascending=False))

# Prepare Data
X = df.drop(['MEDV'], axis=1)
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Crucial for Ridge and Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Models
lr = LinearRegression()
ridge = Ridge(alpha=1.0) # alpha is the regularization strength
lasso = Lasso(alpha=0.1)

# Fit Models
lr.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)

models = {'Linear Regression': lr, 'Ridge': ridge, 'Lasso': lasso}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2 Score: {r2:.4f}\n")

    coeff_df = pd.DataFrame({'Feature': feature_names, 
                         'Linear': lr.coef_, 
                         'Ridge': ridge.coef_, 
                         'Lasso': lasso.coef_})

print("Coefficient Comparison:")
print(coeff_df)

# Plotting Coefficients
coeff_df.set_index('Feature').plot(kind='bar', figsize=(12,6))
plt.title("Comparison of Model Coefficients")
plt.axhline(0, color='black', lw=1)
plt.show()

