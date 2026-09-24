import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("BIKE SHARING DEMAND ANALYTICS AND PREDICTION SYSTEM")
print("=" * 70)

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("hour.csv")

print("\nDATASET SHAPE")
print(df.shape)

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

# --------------------------------------------------
# 2. DATA QUALITY CHECK
# --------------------------------------------------

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nDATASET AFTER REMOVING DUPLICATES")
print(df.shape)

# --------------------------------------------------
# 3. KEY PERFORMANCE INDICATORS
# --------------------------------------------------

total_rentals = df["cnt"].sum()

average_hourly_demand = df["cnt"].mean()

peak_hour = (
    df.groupby("hr")["cnt"]
    .mean()
    .idxmax()
)

peak_hour_demand = (
    df.groupby("hr")["cnt"]
    .mean()
    .max()
)

working_day_demand = (
    df[df["workingday"] == 1]["cnt"].mean()
)

non_working_day_demand = (
    df[df["workingday"] == 0]["cnt"].mean()
)

print("\nKEY PERFORMANCE INDICATORS")
print("-" * 50)

print("Total Rentals:", int(total_rentals))
print("Average Hourly Demand:", round(average_hourly_demand, 2))
print("Peak Demand Hour:", peak_hour)
print("Average Demand at Peak Hour:", round(peak_hour_demand, 2))
print("Average Working-Day Demand:", round(working_day_demand, 2))
print("Average Non-Working-Day Demand:", round(non_working_day_demand, 2))

# --------------------------------------------------
# 4. HOURLY DEMAND
# --------------------------------------------------

hourly_demand = df.groupby("hr")["cnt"].mean()

plt.figure(figsize=(10, 5))
hourly_demand.plot(kind="line", marker="o")
plt.title("Average Bike Rental Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Bike Rentals")
plt.grid(True)
plt.tight_layout()
plt.savefig("hourly_demand.png")
plt.close()

# --------------------------------------------------
# 5. MONTHLY DEMAND
# --------------------------------------------------

monthly_demand = df.groupby("mnth")["cnt"].mean()

plt.figure(figsize=(10, 5))
monthly_demand.plot(kind="bar")
plt.title("Average Bike Rental Demand by Month")
plt.xlabel("Month")
plt.ylabel("Average Bike Rentals")
plt.tight_layout()
plt.savefig("monthly_demand.png")
plt.close()

# --------------------------------------------------
# 6. SEASONAL DEMAND
# --------------------------------------------------

season_demand = df.groupby("season")["cnt"].mean()

plt.figure(figsize=(8, 5))
season_demand.plot(kind="bar")
plt.title("Average Bike Rental Demand by Season")
plt.xlabel("Season")
plt.ylabel("Average Bike Rentals")
plt.tight_layout()
plt.savefig("season_demand.png")
plt.close()

# --------------------------------------------------
# 7. WEATHER DEMAND
# --------------------------------------------------

weather_demand = df.groupby("weathersit")["cnt"].mean()

plt.figure(figsize=(8, 5))
weather_demand.plot(kind="bar")
plt.title("Average Bike Rental Demand by Weather Condition")
plt.xlabel("Weather Situation")
plt.ylabel("Average Bike Rentals")
plt.tight_layout()
plt.savefig("weather_demand.png")
plt.close()

# --------------------------------------------------
# 8. TEMPERATURE VS DEMAND
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="temp",
    y="cnt",
    alpha=0.3
)

plt.title("Temperature vs Bike Rental Demand")
plt.xlabel("Normalized Temperature")
plt.ylabel("Bike Rentals")
plt.tight_layout()
plt.savefig("temperature_vs_demand.png")
plt.close()

# --------------------------------------------------
# 9. HUMIDITY VS DEMAND
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="hum",
    y="cnt",
    alpha=0.3
)

plt.title("Humidity vs Bike Rental Demand")
plt.xlabel("Normalized Humidity")
plt.ylabel("Bike Rentals")
plt.tight_layout()
plt.savefig("humidity_vs_demand.png")
plt.close()

# --------------------------------------------------
# 10. WORKING DAY VS NON-WORKING DAY
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="workingday",
    y="cnt"
)

plt.title("Bike Demand: Working Days vs Non-Working Days")
plt.xlabel("Working Day (0 = No, 1 = Yes)")
plt.ylabel("Bike Rentals")
plt.tight_layout()
plt.savefig("workingday_demand.png")
plt.close()

# --------------------------------------------------
# 11. CORRELATION ANALYSIS
# --------------------------------------------------

numeric_columns = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "cnt"
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# --------------------------------------------------
# 12. MACHINE LEARNING
# --------------------------------------------------

features = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed"
]

X = df[features]

y = df["cnt"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# --------------------------------------------------
# 13. MODEL EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 50)

print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# --------------------------------------------------
# 14. FEATURE IMPORTANCE
# --------------------------------------------------

feature_importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFEATURE IMPORTANCE")
print("-" * 50)

print(feature_importance)

plt.figure(figsize=(10, 6))

feature_importance.sort_values().plot(
    kind="barh"
)

plt.title("Factors Influencing Bike Rental Demand")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# --------------------------------------------------
# 15. SAMPLE PREDICTIONS
# --------------------------------------------------

results = pd.DataFrame({
    "Actual Demand": y_test.values,
    "Predicted Demand": y_pred
})

print("\nSAMPLE PREDICTIONS")
print(results.head(10))

# --------------------------------------------------
# 16. BUSINESS INSIGHTS
# --------------------------------------------------

print("\nKEY INSIGHTS")
print("-" * 50)

print(
    "1. Bike rental demand varies significantly across different hours."
)

print(
    "2. Seasonal conditions influence overall bike rental demand."
)

print(
    "3. Weather conditions provide useful information for demand planning."
)

print(
    "4. Temperature and humidity are associated with changes in rental demand."
)

print(
    "5. Time-related features such as hour and month are important for demand prediction."
)

# --------------------------------------------------
# 17. RISKS
# --------------------------------------------------

print("\nRISKS")
print("-" * 50)

print(
    "1. Unfavorable weather can reduce bike rental demand."
)

print(
    "2. Unexpected demand spikes may create bike shortages."
)

print(
    "3. Poor demand forecasting can result in inefficient bike redistribution."
)

# --------------------------------------------------
# 18. OPPORTUNITIES
# --------------------------------------------------

print("\nOPPORTUNITIES")
print("-" * 50)

print(
    "1. Demand forecasting can improve bike redistribution."
)

print(
    "2. Peak-hour forecasting can improve bike availability."
)

print(
    "3. Weather-aware planning can improve operational efficiency."
)

print(
    "4. Historical demand patterns can support resource planning."
)

# --------------------------------------------------
# 19. RECOMMENDED ACTIONS
# --------------------------------------------------

print("\nRECOMMENDED ACTIONS")
print("-" * 50)

print(
    "1. Increase bike availability during predicted high-demand periods."
)

print(
    "2. Use weather information when planning bike distribution."
)

print(
    "3. Monitor peak-hour demand to reduce shortages."
)

print(
    "4. Use predictive analytics to support daily redistribution decisions."
)

print(
    "5. Continuously monitor demand patterns and update forecasts."
)

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)