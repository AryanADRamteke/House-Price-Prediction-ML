import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv("AmesHousing.csv")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

# Check missing values
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False))

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

print("Number of Numerical Columns:", len(numerical_cols))
print("Number of Categorical Columns:", len(categorical_cols))

# Statistical summary
print(df.describe())
print(df['SalePrice'].describe())

# ---------------------------------------------------------
# Visual exploration
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(df['SalePrice'], bins=30)
plt.title("Distribution of House Prices")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.show()

# Correlation with Sale Price
correlation = df.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)
print(correlation.head(15))

plt.figure(figsize=(10, 6))
plt.scatter(df["Gr Liv Area"], df["SalePrice"])
plt.xlabel("Ground Living Area")
plt.ylabel("Sale Price")
plt.title("Ground Living Area vs Sale Price")
plt.show()

# ---------------------------------------------------------
# Feature selection + train/test split
# ---------------------------------------------------------
X = df[['Overall Qual', 'Gr Liv Area', 'Garage Cars',
        'Garage Area', 'Total Bsmt SF', '1st Flr SF']]
y = df['SalePrice']

# Fill missing values BEFORE splitting, so train and test are consistent
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)

# ---------------------------------------------------------
# Linear Regression
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")

y_pred = model.predict(X_test)

print("R2 Score :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))

# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest R2:", r2_score(y_test, rf_pred))
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})
importance = importance.sort_values(by="Importance", ascending=False)
print(importance)

plt.figure(figsize=(8, 5))
plt.bar(importance["Feature"], importance["Importance"])
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Decision Tree R2:", r2_score(y_test, dt_pred))
print("Decision Tree MAE:", mean_absolute_error(y_test, dt_pred))

# ---------------------------------------------------------
# Compare Models
# ---------------------------------------------------------
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "R2 Score": [
        r2_score(y_test, y_pred),
        r2_score(y_test, dt_pred),
        r2_score(y_test, rf_pred)
    ],
    "MAE": [
        mean_absolute_error(y_test, y_pred),
        mean_absolute_error(y_test, dt_pred),
        mean_absolute_error(y_test, rf_pred)
    ]
})
print(comparison.sort_values(by="R2 Score", ascending=False))

# ---------------------------------------------------------
# Hyperparameter Tuning (Random Forest)
# ---------------------------------------------------------
params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best R2 Score:", grid.best_score_)

best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)

print("Final R2:", r2_score(y_test, best_pred))
print("Final MAE:", mean_absolute_error(y_test, best_pred))

# ---------------------------------------------------------
# Save + load model
# ---------------------------------------------------------
joblib.dump(best_model, "house_price_model.pkl")
print("Model saved successfully!")

loaded_model = joblib.load("house_price_model.pkl")

prediction = loaded_model.predict(X_test.iloc[[0]])
print("Predicted Price:", prediction[0])
print("Actual Price:", y_test.iloc[0])

# ---------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------
def predict_house_price(overall_qual, gr_liv_area, garage_cars,
                         garage_area, total_bsmt_sf, first_flr_sf):
    """Predict sale price for a single house given its key features.

    Uses a DataFrame with proper column names (matching training data)
    so the model doesn't throw a missing-feature-names warning.
    """
    new_house = pd.DataFrame({
        'Overall Qual': [overall_qual],
        'Gr Liv Area': [gr_liv_area],
        'Garage Cars': [garage_cars],
        'Garage Area': [garage_area],
        'Total Bsmt SF': [total_bsmt_sf],
        '1st Flr SF': [first_flr_sf]
    })

    price = loaded_model.predict(new_house)
    return price[0]


predicted_price = predict_house_price(
    8,      # Overall Quality
    2200,   # Ground Living Area
    2,      # Garage Cars
    500,    # Garage Area
    900,    # Total Basement SF
    1200    # First Floor SF
)

print("Predicted House Price:", predicted_price)
