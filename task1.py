import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = pd.read_csv(r'D:\python\house_prices.csv')
print("Dataset loaded successfully!")
print(f"Columns: {data.columns.tolist()}")

# Prepare features - using columns that exist in your dataset
features = ['Carpet Area', 'Bathroom', 'Balcony', 'Car Parking']
target = 'Price (in rupees)'

# Convert columns to numeric (handling commas, spaces, etc.)
for col in features + [target]:
    # Clean and convert to numeric
    data[col] = pd.to_numeric(
        data[col].astype(str).str.replace(',', '').str.replace(' ', ''),
        errors='coerce'
    )

# Fill missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Prepare features and target
X = data[features].copy()
y = data[target]

print("\nSample data:")
print(X.head())
print(f"\nTarget values:\n{y.head()}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f'\nRMSE: {rmse:.2f}')
print(f'R² Score: {r2:.4f}')

# Predict on new data
# Example values: [Carpet Area, Bathroom, Balcony, Car Parking]
new_data = [1200, 2, 2, 1]  # Adjust these values as needed
input_df = pd.DataFrame([new_data], columns=features)
predicted_price = model.predict(input_df)[0]
print(f'\nPredicted Price: ₹{predicted_price:,.2f}')
