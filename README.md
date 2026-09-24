# Bike Sharing Demand Analytics and Prediction System

## 1. Project Overview

The Bike Sharing Demand Analytics and Prediction System analyzes historical hourly bike rental data to identify demand patterns, understand the factors affecting bike rentals, and predict bike rental demand using machine learning.

The project converts raw bike-sharing data into key performance indicators, trends, demand drivers, risks, opportunities, and recommended actions.

## 2. Problem Statement

Bike-sharing systems experience changes in demand depending on time, season, weather conditions, working days, and other environmental factors.

Accurate demand analysis and prediction can help bike-sharing operators improve bike availability, plan redistribution, and make data-driven operational decisions.

This project analyzes historical bike-sharing data and develops a machine learning model to predict hourly bike rental demand.

## 3. Objectives

* Analyze historical bike rental demand.
* Identify peak demand hours.
* Analyze demand across different months and seasons.
* Study the relationship between weather conditions and bike demand.
* Compare working-day and non-working-day demand.
* Identify important factors affecting demand.
* Build a machine learning model for demand prediction.
* Generate actionable recommendations.

## 4. Dataset

Dataset: Bike Sharing Dataset

Source: Kaggle

Dataset Link: https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset

The project uses the `hour.csv` dataset.

The dataset contains hourly information about bike rentals and variables related to time, season, weather, temperature, humidity, windspeed, working days, and rental counts.

The target variable is `cnt`, representing the total number of bike rentals.

The variables `casual` and `registered` are not used as prediction features because they are components of the total rental count.

## 5. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## 6. Key Performance Indicators

The project calculates:

* Total bike rentals
* Average hourly demand
* Peak demand hour
* Average demand during the peak hour
* Average working-day demand
* Average non-working-day demand

## 7. Methodology

The project follows these steps:

1. Data collection
2. Data loading
3. Data quality checking
4. Duplicate removal
5. Exploratory data analysis
6. Data visualization
7. Correlation analysis
8. Feature selection
9. Train-test splitting
10. Machine learning model training
11. Model evaluation
12. Feature importance analysis
13. Insight generation
14. Risk and opportunity identification
15. Recommended actions

## 8. Data Analysis

The project analyzes bike demand based on:

* Hour of the day
* Month
* Season
* Weather condition
* Temperature
* Humidity
* Working day status

The project generates visualizations including:

* Hourly demand
* Monthly demand
* Seasonal demand
* Weather-based demand
* Temperature versus demand
* Humidity versus demand
* Working-day versus non-working-day demand
* Correlation heatmap
* Feature importance

## 9. Machine Learning Model

A Random Forest Regressor is used to predict total hourly bike rental demand.

The target variable is:

`cnt`

The model uses:

* Season
* Year
* Month
* Hour
* Holiday
* Weekday
* Working day
* Weather situation
* Temperature
* Feels-like temperature
* Humidity
* Windspeed

The dataset is divided into training and testing sets using an 80:20 split.

## 10. Model Evaluation

The model is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The actual evaluation results are generated when the Python program is executed.

## 11. Risks

Potential operational risks identified include:

* Reduced demand during unfavorable weather.
* Unexpected demand spikes.
* Insufficient bike availability during high-demand periods.
* Inefficient redistribution when demand is not forecast accurately.

## 12. Opportunities

The analysis provides opportunities to:

* Improve bike redistribution.
* Prepare for peak demand periods.
* Incorporate weather information into operational planning.
* Improve bike availability.
* Use predictive analytics for resource planning.

## 13. Recommended Actions

* Increase bike availability during predicted high-demand periods.
* Use weather information when planning bike distribution.
* Monitor peak-hour demand to reduce shortages.
* Use predictive analytics to support daily redistribution decisions.
* Continuously monitor demand patterns and update forecasts.

## 14. How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python bike_sharing_analysis.py
```

## 15. Project Outputs

The project generates:

* Hourly demand analysis
* Monthly demand analysis
* Seasonal demand analysis
* Weather-based demand analysis
* Temperature and humidity analysis
* Working-day analysis
* Correlation analysis
* Feature importance analysis
* Machine learning predictions
* Model evaluation metrics

## 16. Conclusion

The project demonstrates how bike-sharing data can be transformed into meaningful business intelligence. Through data analytics, visualization, and machine learning, the system identifies demand patterns and provides actionable recommendations for improving bike availability, redistribution, and operational planning.
