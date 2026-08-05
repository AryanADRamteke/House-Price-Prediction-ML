import pandas as pd

# Load dataset
df = pd.read_csv("AmesHousing.csv")

# Display first 5 rows
print(df.head())

# Dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

# Check missing values
missing_values = df.isnull().sum()

# Display columns with missing values
print(missing_values[missing_values > 0].sort_values(ascending=False))

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'string']).columns

print("Number of Numerical Columns:", len(numerical_cols))
print("Number of Categorical Columns:", len(categorical_cols))

# Statistical summary
print(df.describe())

print(df['SalePrice'].describe())

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.hist(df['SalePrice'], bins=30)
plt.title("Distribution of House Prices")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.show()

# Correlation with Sale Price
correlation = df.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)

print(correlation.head(15))

plt.figure(figsize=(10,6))
plt.scatter(df["Gr Liv Area"], df["SalePrice"])
plt.xlabel("Ground Living Area")
plt.ylabel("Sale Price")
plt.title("Ground Living Area vs Sale Price")
plt.show()

from sklearn.model_selection import train_test_split

# Select important features
X = df[['Overall Qual', 'Gr Liv Area', 'Garage Cars',
        'Garage Area', 'Total Bsmt SF', '1st Flr SF']]

y = df['SalePrice']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

X = X.fillna(X.median())

from sklearn.linear_model import LinearRegression

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

X = X.fillna(X.median())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

from sklearn.metrics import r2_score, mean_absolute_error

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score :", r2_score(y_test, y_pred))
print("MAE :", mean_absolute_error(y_test, y_pred))

from sklearn.ensemble import RandomForestRegressor

# Train Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Predict
rf_pred = rf.predict(X_test)

# Evaluate
print("Random Forest R2:", r2_score(y_test, rf_pred))
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))

import pandas as pd

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(by="Importance", ascending=False)

print(importance)

import matplotlib.pyplot as plt

# Plot Feature Importance
plt.figure(figsize=(8,5))
plt.bar(importance["Feature"], importance["Importance"])
plt.xticks(rotation=45)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

from sklearn.tree import DecisionTreeRegressor

# Train Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Predict
dt_pred = dt.predict(X_test)

# Evaluate
print("Decision Tree R2:", r2_score(y_test, dt_pred))
print("Decision Tree MAE:", mean_absolute_error(y_test, dt_pred))

# Compare Models
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

from sklearn.model_selection import GridSearchCV

# Hyperparameter Tuning
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

import joblib

# Save model
joblib.dump(best_model, "house_price_model.pkl")

print("Model saved successfully!")

# Load saved model
loaded_model = joblib.load("house_price_model.pkl")

# Test prediction
prediction = loaded_model.predict(X_test.iloc[[0]])

print("Predicted Price:", prediction[0])
print("Actual Price:", y_test.iloc[0])

def predict_house_price(overall_qual, gr_liv_area, garage_cars,
                        garage_area, total_bsmt_sf, first_flr_sf):

    data = [[overall_qual, gr_liv_area, garage_cars,
             garage_area, total_bsmt_sf, first_flr_sf]]

    price = loaded_model.predict(data)

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

def predict_house_price(overall_qual, gr_liv_area, garage_cars,
                        garage_area, total_bsmt_sf, first_flr_sf):

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
    8,
    2200,
    2,
    500,
    900,
    1200
)

print("Predicted House Price:", predicted_price)