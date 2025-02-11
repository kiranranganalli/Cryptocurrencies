#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:17:49 2025

@author: kiran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
file_path = "currencies_data.csv"  # Update this with your actual file path
crypto_df = pd.read_csv(file_path)

# Display basic info
print(crypto_df.info())
print(crypto_df.head())

# Check missing values
print("Missing Values:\n", crypto_df.isnull().sum())

crypto_df['maxSupply'].fillna(0, inplace=True)
crypto_df['maxSupply'].fillna(crypto_df['maxSupply'].median(), inplace=True)
crypto_df.drop(columns=['maxSupply'], inplace=True)

# Remove duplicate rows
crypto_df.drop_duplicates(inplace=True)

# Verify duplicate removal
print(f"Remaining duplicate rows: {crypto_df.duplicated().sum()}")

# Drop redundant columns
crypto_df.drop(columns=['name.1'], inplace=True)

# If `isActive` has only 1 value (all active), drop it
if crypto_df['isActive'].nunique() == 1:
    crypto_df.drop(columns=['isActive'], inplace=True)

# Check remaining columns
print("Updated Columns:", crypto_df.columns)

# Save the cleaned dataset
crypto_df.to_csv("cleaned_crypto_data.csv", index=False)
print("Cleaned dataset saved successfully!")

###############################################

# Load the cleaned dataset
cleaned_file_path = "cleaned_crypto_data.csv"  # Ensure this is the correct file
crypto_df = pd.read_csv(cleaned_file_path)

# Select top 10 ranked cryptocoins
top_10_ranked = crypto_df.nsmallest(10, 'cmcRank')[['cmcRank', 'name', 'marketCap', 'price']]

# Create the plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    x=top_10_ranked['name'], 
    y=top_10_ranked['marketCap'], 
    palette="Blues_r"
)

# Display values on top of bars
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():,.0f}",  # Format numbers with commas
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', va='bottom', 
        fontsize=10, fontweight='bold', color='black'
    )

# Labels and Title
plt.xticks(rotation=45, ha='right')
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("Market Capitalization (in USD)", fontsize=12)
plt.title("Top 5 Highest Ranked Cryptocoins", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show the plot
plt.show()

####################################################

# Convert date columns to datetime format
crypto_df['dateAdded'] = pd.to_datetime(crypto_df['dateAdded'])
crypto_df['lastUpdated'] = pd.to_datetime(crypto_df['lastUpdated'])

# Get today's date
today = datetime.today()

# Calculate the age of each cryptocurrency
crypto_df['age_days_since_added'] = (today - crypto_df['dateAdded']).dt.days
crypto_df['age_days_since_updated'] = (today - crypto_df['lastUpdated']).dt.days

# Get the Top 10 Oldest Cryptocoins based on dateAdded
top_10_oldest = crypto_df.nlargest(10, 'age_days_since_added')[['name', 'symbol', 'dateAdded', 'age_days_since_added']]

# Display the result
print("Top 10 Oldest Cryptocoins:")
print(top_10_oldest)


# Ensure unique cryptocurrencies (sorted by age)
top_10_oldest = crypto_df.sort_values(by='age_days_since_added', ascending=False).drop_duplicates(subset=['name']).head(10)

# Convert days to years
top_10_oldest['age_years'] = top_10_oldest['age_days_since_added'] / 365

# Create the plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=top_10_oldest, 
    x="name", 
    y="age_years", 
    palette="Purples_r"
)

# Display values on top of bars
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.1f} years",  # Display 1 decimal place
        (p.get_x() + p.get_width() / 2., p.get_height()), 
        ha='center', va='bottom', 
        fontsize=10, fontweight='bold', color='black'
    )

# Add a legend indicating the dataset reference year
plt.legend(["As of 2023"], loc="upper right", frameon=True, fontsize=12, edgecolor="black")

# Labels and Title
plt.xticks(rotation=45, ha='right')
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("Age (Years Since Added)", fontsize=12)
plt.title("Top 10 Oldest Cryptocoins", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show the plot
plt.show()

#############################################################


# Select the top 10 cryptocurrencies based on market cap
top_10_prices = crypto_df.nsmallest(10, 'cmcRank')[['name', 'price']]

# Create a bar chart
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=top_10_prices, x='name', y='price', palette='Blues_r')

# Display values on top of bars
for p in ax.patches:
    ax.annotate(f"${p.get_height():,.2f}", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Labels and Title
plt.xticks(rotation=45, ha='right')
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.title("Price Comparison of Top 10 Cryptocoins", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show the plot
plt.show()

################################################################

# Select the top 10 cryptocurrencies based on market cap
top_10_market_cap = crypto_df.nlargest(10, 'marketCap')[['name', 'marketCap', 'marketCapByTotalSupply']]

# Check the difference between the two metrics
top_10_market_cap['cap_difference'] = top_10_market_cap['marketCapByTotalSupply'] - top_10_market_cap['marketCap']

# Display data
print(top_10_market_cap)

# Set figure size
plt.figure(figsize=(12, 6))

# Create grouped bar chart
top_10_market_cap.plot(
    x="name",
    kind="bar",
    stacked=False,
    figsize=(12, 6),
    color=["royalblue", "orange"],
    width=0.8
)

# Labels and title
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("Market Cap (USD)", fontsize=12)
plt.title("Market Cap vs. Market Cap by Total Supply", fontsize=14)
plt.xticks(rotation=45, ha="right")

# Add legend
plt.legend(["Market Cap", "Market Cap by Total Supply"], loc="upper left", fontsize=10)

# Show grid
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show the plot
plt.show()

################################################################

# Select the top 10 cryptocurrencies based on market cap
top_10_change = crypto_df.nlargest(10, 'marketCap')[
    ['name', 'percentChange1h', 'percentChange24h', 'percentChange7d', 
     'percentChange30d', 'percentChange60d', 'percentChange90d']
]

# Define the time frames
time_frames = {
    "1 Hour": "percentChange1h",
    "24 Hours": "percentChange24h",
    "7 Days": "percentChange7d",
    "30 Days": "percentChange30d",
    "60 Days": "percentChange60d",
    "90 Days": "percentChange90d"
}

# Create separate bar charts for each time frame
for time_label, column in time_frames.items():
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=top_10_change, x="name", y=column, palette="coolwarm")

    # Display values on top of bars
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}%", 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='black'
        )

    # Labels and title
    plt.xlabel("Cryptocurrency", fontsize=12)
    plt.ylabel("Percentage Change (%)", fontsize=12)
    plt.title(f"Cryptocurrency Percentage Change - {time_label}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    
    # Show grid
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Show the plot
    plt.show()
    
########################################################
    
# Select the top 15 cryptocurrencies based on market cap
top_15_dominance = crypto_df.nlargest(15, 'marketCap')[['name', 'marketCap']]

# Calculate percentage share
top_15_dominance['marketCapPercentage'] = (top_15_dominance['marketCap'] / top_15_dominance['marketCap'].sum()) * 100

# Display data
print(top_15_dominance)


# Create the pie chart
plt.figure(figsize=(10, 6))
plt.pie(
    top_15_dominance['marketCapPercentage'], 
    labels=top_15_dominance['name'], 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=plt.cm.Paired.colors,
    wedgeprops={'edgecolor': 'black'}
)

# Title and display
plt.title("Market Dominance of Top 15 Cryptocurrencies", fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()

######################################################

# Select the top 15 cryptocurrencies based on market cap
top_15_ytd = crypto_df.nlargest(15, 'marketCap')[['name', 'ytdPriceChangePercentage']]

# Display data
print(top_15_ytd)

# Set figure size
plt.figure(figsize=(12, 6))

# Create the bar chart
ax = sns.barplot(data=top_15_ytd, x='name', y='ytdPriceChangePercentage', palette="coolwarm")

# Display values on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.1f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Labels and title
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("YTD Price Change (%)", fontsize=12)
plt.title("YTD Price Change Comparison for Top 8 Cryptocurrencies", fontsize=14)
plt.xticks(rotation=45, ha="right")

# Show grid
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show the plot
plt.show()

######################################################

# Select relevant columns for volatility analysis
volatility_columns = ['percentChange1h', 'percentChange24h', 'percentChange7d', 
                      'percentChange30d', 'percentChange60d', 'percentChange90d']

# Calculate standard deviation for each cryptocurrency and convert to percentage
crypto_df['volatility_percentage'] = crypto_df[volatility_columns].std(axis=1) * 100

# Select the top 10 most volatile cryptocurrencies
top_10_volatile = crypto_df.nlargest(10, 'volatility_percentage')[['name', 'volatility_percentage']]

# Display data
print(top_10_volatile)

# Set figure size
plt.figure(figsize=(12, 6))

# Create the bar chart
ax = sns.barplot(data=top_10_volatile, x='name', y='volatility_percentage', palette="coolwarm")

# Display values on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%",  # Show percentage format
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Labels and title
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("Volatility (%)", fontsize=12)
plt.title("Top 10 Most Volatile Cryptocurrencies", fontsize=14)
plt.xticks(rotation=45, ha="right")

# Show grid
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show the plot
plt.show()

#---------------------------------------------------------------

# Select relevant columns for volatility analysis
volatility_columns = ['percentChange1h', 'percentChange24h', 'percentChange7d', 
                      'percentChange30d', 'percentChange60d', 'percentChange90d']

# Calculate standard deviation (volatility) for each cryptocurrency and convert to percentage
crypto_df['volatility_percentage'] = crypto_df[volatility_columns].std(axis=1) * 100

# Select the Top 10 cryptocurrencies by Market Cap
top_10_market_cap_volatility = crypto_df.nlargest(10, 'marketCap')[['name', 'volatility_percentage']]

# Display data
print(top_10_market_cap_volatility)

# Set figure size
plt.figure(figsize=(12, 6))

# Create the bar chart
ax = sns.barplot(data=top_10_market_cap_volatility, x='name', y='volatility_percentage', palette="coolwarm")

# Display values on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%",  # Show percentage format
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Labels and title
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("Volatility (%)", fontsize=12)
plt.title("Volatility of Top 10 Cryptocurrencies by Market Cap", fontsize=14)
plt.xticks(rotation=45, ha="right")

# Show grid
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show the plot
plt.show()

##############################################################################

# Select the top 10 cryptocurrencies based on 24-hour trading volume
top_10_volume = crypto_df.nlargest(10, 'volume24h')[['name', 'volume24h']]

# Display data
print(top_10_volume)

import matplotlib.pyplot as plt
import seaborn as sns

# Set figure size
plt.figure(figsize=(12, 6))

# Create the bar chart
ax = sns.barplot(data=top_10_volume, x='name', y='volume24h', palette="Blues_r")

# Display values on top of bars
for p in ax.patches:
    ax.annotate(f"${p.get_height():,.0f}",  
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Labels and title
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("Trading Volume (24h) in USD", fontsize=12)
plt.title("Top 10 Cryptocurrencies by 24H Trading Volume", fontsize=14)
plt.xticks(rotation=45, ha="right")

# Show grid
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show the plot
plt.show()

#############################################################################

# Select relevant columns for correlation analysis
correlation_columns = ['marketCap', 'percentChange1h', 'percentChange24h', 
                       'percentChange7d', 'percentChange30d', 'percentChange60d', 'percentChange90d']

# Compute correlation matrix
correlation_matrix = crypto_df[correlation_columns].corr()

# Display correlation values
print(correlation_matrix)

# Set figure size
plt.figure(figsize=(10, 6))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")

# Labels and title
plt.title("Correlation Between Market Cap and Price Changes", fontsize=14)
plt.xlabel("Price Change Timeframes", fontsize=12)
plt.ylabel("Metrics", fontsize=12)

# Show the plot
plt.show()

###########################################################################

# Select the top 100 cryptocurrencies by market cap to ensure variety
top_crypto = crypto_df.nlargest(100, 'marketCap').copy()

# Select relevant columns for price fluctuation
volatility_columns = ['percentChange1h', 'percentChange24h', 'percentChange7d', 
                      'percentChange30d', 'percentChange60d', 'percentChange90d', 'ytdPriceChangePercentage']

# Replace missing values with 0 to avoid errors
top_crypto[volatility_columns] = top_crypto[volatility_columns].fillna(0)

# Compute the mean absolute deviation (better for stability measurement)
top_crypto['stability_score'] = top_crypto[volatility_columns].abs().mean(axis=1)

# Remove coins with zero volatility to avoid missing data
top_crypto = top_crypto[top_crypto['stability_score'] > 0]

# Remove duplicate coin names while keeping the most stable entry
top_crypto = top_crypto.sort_values(by=['stability_score']).drop_duplicates(subset=['name'])

# Ensure exactly 5 stable cryptocurrencies are selected, adjusting range if needed
top_5_stable_market_cap = top_crypto.nsmallest(5, 'stability_score')[['name', 'stability_score']]

# Display data
print(top_5_stable_market_cap)

# Set figure size
plt.figure(figsize=(10, 6))

# Create a bar chart for stability ranking
ax = sns.barplot(data=top_5_stable_market_cap, x='name', y='stability_score', palette="Blues_r")

# Display correctly formatted values on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.6f}",  
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Labels and title
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("Stability Score (Lower is More Stable)", fontsize=12)
plt.title("Top 5 Most Stable Cryptocurrencies (Within Top 100 by Market Cap)", fontsize=14)
plt.xticks(rotation=45, ha="right")

# Show grid
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show the plot
plt.show()

#############################################################################

# Select the top 200 cryptocurrencies by market cap to ensure a broad selection
top_crypto = crypto_df.nlargest(200, 'marketCap').copy()

# Select relevant columns for growth analysis
growth_columns = ['percentChange7d', 'percentChange30d', 'percentChange60d', 'percentChange90d']

# Replace missing values with 0
top_crypto[growth_columns] = top_crypto[growth_columns].fillna(0)

# Compute the overall average growth rate across all timeframes
top_crypto['growth_rate'] = top_crypto[growth_columns].mean(axis=1)

# Remove duplicate coin names while keeping the highest growth rate entry for each unique cryptocurrency
top_crypto = top_crypto.sort_values(by=['growth_rate'], ascending=False).drop_duplicates(subset=['name'])

# Ensure exactly 10 cryptocurrencies are selected
top_10_growth_market_cap = top_crypto.head(10)[['name', 'growth_rate']]

# Display data
print(top_10_growth_market_cap)

# Set figure size
plt.figure(figsize=(12, 6))

# Create a bar chart for growth ranking
ax = sns.barplot(data=top_10_growth_market_cap, x='name', y='growth_rate', palette="Greens_r")

# Display correctly formatted values on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%",  
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Labels and title
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("Average Growth Rate (%)", fontsize=12)
plt.title("Top 10 Fastest-Growing Cryptocurrencies (Within Top 200 Market Cap)", fontsize=14)
plt.xticks(rotation=45, ha="right")

# Show grid
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Show the plot
plt.show()

##############################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
file_path = "cleaned_crypto_data.csv"
crypto_df = pd.read_csv(file_path)

# Select features for prediction
features = ['marketCap', 'volume24h', 'marketCapByTotalSupply', 'dominance',
            'percentChange1h', 'percentChange24h', 'percentChange7d', 'percentChange30d', 
            'percentChange60d', 'percentChange90d', 'ytdPriceChangePercentage']

target = 'price'  # Predicting price

# Drop rows with missing values
crypto_df = crypto_df.dropna(subset=features + [target])

# Keep Coin Name and Symbol for reference
coin_names = crypto_df[['name', 'symbol']].reset_index(drop=True)

# Define X (features) and y (target variable)
X = crypto_df[features]
y = np.log1p(crypto_df[target])  # Apply log transformation for better prediction accuracy

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test, coin_train, coin_test = train_test_split(X_scaled, y, coin_names, test_size=0.2, random_state=42)

# Train XGBoost model (Best performing model)
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and reverse log transformation
y_pred_test = np.expm1(xgb_model.predict(X_test))

# Create DataFrame with Coin Name, Actual Price, and Predicted Price
predictions_df = pd.DataFrame({
    'Coin Name': coin_test['name'].values,
    'Symbol': coin_test['symbol'].values,
    'Actual Price': np.expm1(y_test).values,
    'Predicted Price': y_pred_test
})

# Save the predictions to a CSV file
predictions_df.to_csv("final_price_predictions.csv", index=False)

# Display first few predictions
print(predictions_df.head())

###########################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the predictions file
file_path = "final_price_predictions.csv"
predictions_df = pd.read_csv(file_path)

# Ensure correct column names
predictions_df = predictions_df[['Coin Name', 'Symbol', 'Actual Price', 'Predicted Price']]

# Sort by actual price to get the top 10 highest value coins
top_10_predictions = predictions_df.sort_values(by="Actual Price", ascending=False).head(10)

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=top_10_predictions, x="Coin Name", y="Actual Price", color="blue", label="Actual Price")
sns.barplot(data=top_10_predictions, x="Coin Name", y="Predicted Price", color="orange", label="Predicted Price")

# Add values on top of bars
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}",  
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

# Labels and title
plt.xlabel("Cryptocurrency", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.title("Actual vs Predicted Price for Top 10 Cryptocurrencies", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.legend()

# Show the plot
plt.show()

#############################################################################

import pandas as pd

# Load dataset
file_path = "cleaned_crypto_data.csv"
df = pd.read_csv(file_path)

# Ensure no zero or missing prices
df = df.dropna(subset=['price'])

# Calculate historical prices using reverse percentage formula
df['price_1h_ago'] = df['price'] / (1 + (df['percentChange1h'] / 100))
df['price_24h_ago'] = df['price'] / (1 + (df['percentChange24h'] / 100))
df['price_7d_ago'] = df['price'] / (1 + (df['percentChange7d'] / 100))
df['price_30d_ago'] = df['price'] / (1 + (df['percentChange30d'] / 100))
df['price_60d_ago'] = df['price'] / (1 + (df['percentChange60d'] / 100))
df['price_90d_ago'] = df['price'] / (1 + (df['percentChange90d'] / 100))

# Save updated dataset
df.to_csv("historical_crypto_prices.csv", index=False)

# Display first few rows
print(df[['name', 'symbol', 'price', 'price_1h_ago', 'price_24h_ago', 'price_7d_ago', 'price_30d_ago', 'price_60d_ago', 'price_90d_ago']].head())

#----------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet  # Instead of "from fbprophet import Prophet"

# Load the dataset with historical prices
file_path = "historical_crypto_prices.csv"  # Ensure correct path
df = pd.read_csv(file_path)

# Select a cryptocurrency for forecasting (e.g., Bitcoin)
coin_name = "Bitcoin"  # Change this if needed
coin_df = df[df['name'] == coin_name].copy()

# Ensure data is sorted by time
coin_df = coin_df.sort_values(by="lastUpdated")

# Prepare data for ARIMA (Short-term forecasting)
arima_data = coin_df[['lastUpdated', 'price']].dropna()
arima_data.set_index("lastUpdated", inplace=True)

# Convert index to datetime for ARIMA
arima_data.index = pd.to_datetime(arima_data.index)

# Train ARIMA with new differenced data
arima_model = ARIMA(coin_df['price_diff'].dropna(), order=(1,1,1))  # Adjust p,d,q
arima_fit = arima_model.fit()

# Forecast
forecast_steps = 30
arima_forecast = arima_fit.forecast(steps=forecast_steps)

# Prepare data for Prophet model (Alternative short-term forecasting)
prophet_df = coin_df[['lastUpdated', 'price']].rename(columns={"lastUpdated": "ds", "price": "y"})
prophet_df.dropna(inplace=True)

# Convert dates for Prophet
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

# Train Prophet model
prophet_model = Prophet()
prophet_model.fit(prophet_df)

# Create future dates for next 30 days
future = prophet_model.make_future_dataframe(periods=30)
forecast = prophet_model.predict(future)

# Plot ARIMA Forecast
plt.figure(figsize=(12, 6))
plt.plot(arima_data, label="Actual Prices", color="blue")
plt.plot(arima_forecast, label="ARIMA Forecast (Next 30 Days)", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title(f"Short-Term Price Prediction for {coin_name} (ARIMA)")
plt.legend()
plt.show()

# Plot Prophet Forecast
plt.figure(figsize=(12, 6))
plt.plot(prophet_df["ds"], prophet_df["y"], label="Actual Prices", color="blue")
plt.plot(forecast["ds"], forecast["yhat"], label="Prophet Forecast (Next 30 Days)", linestyle="dashed", color="green")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title(f"Short-Term Price Prediction for {coin_name} (Prophet)")
plt.legend()
plt.show()



########################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_path = "historical_crypto_prices.csv"
df = pd.read_csv(file_path)

# Select a cryptocurrency (e.g., Bitcoin)
coin_name = "Bitcoin"
coin_df = df[df['name'] == coin_name].copy()

# Sort data by date
coin_df = coin_df.sort_values(by="lastUpdated")

# Convert date column to datetime format
coin_df['lastUpdated'] = pd.to_datetime(coin_df['lastUpdated'])

# Select only the 'price' column for forecasting
data = coin_df[['lastUpdated', 'price']].dropna()

# Check if we have at least 2 data points
if len(data) < 2:
    raise ValueError("Not enough data for forecasting. At least 2 data points are required.")

# Compute Linear Trend (Using Last Two Known Data Points)
x_values = np.arange(len(data))  # Time index
y_values = data['price'].values  # Prices

# Fit a simple linear trend (y = mx + c)
m, c = np.polyfit(x_values, y_values, 1)  # Fit linear regression

# Predict next 5 days
future_days = 5
future_x = np.arange(len(data), len(data) + future_days)
future_y = m * future_x + c  # Linear trend projection

# Create future dataframe
future_dates = pd.date_range(start=data['lastUpdated'].iloc[-1], periods=future_days + 1, freq='D')[1:]
future_df = pd.DataFrame({'lastUpdated': future_dates, 'Predicted Price': future_y})

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 6))
plt.plot(data['lastUpdated'], data['price'], label="Actual Prices", color="blue")
plt.plot(future_df['lastUpdated'], future_df['Predicted Price'], label="Linear Forecast (Next 5 Days)", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title(f"Price Prediction for {coin_name} (Linear Projection)")
plt.legend()
plt.show()

# Display predicted prices
print(future_df)






import numpy as np
import pandas as pd

# Load dataset
file_path = "historical_crypto_prices.csv"
df = pd.read_csv(file_path)

# Replace 'inf' values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values in price columns
price_columns = ['price', 'price_1h_ago', 'price_24h_ago', 'price_7d_ago', 
                 'price_30d_ago', 'price_60d_ago', 'price_90d_ago']
df.dropna(subset=price_columns, inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_crypto_prices.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_crypto_prices.csv'")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load cleaned dataset
file_path = "cleaned_crypto_prices.csv"
df = pd.read_csv(file_path)

# Select the top 10 coins based on market cap
top_10_coins = ["Bitcoin", "Ethereum", "Tether USDt", "BNB", "XRP", "USD Coin",
                "Dora Factory (new)", "Lido Staked ETH", "Cardano"]

# Features (market-based factors)
features = ['marketCap', 'volume24h', 'percentChange24h']

target = 'price'  # Predicting the current price

# Store predictions
forecast_results = {}

# Loop through each top coin and predict prices
for coin in top_10_coins:
    coin_df = df[df['name'] == coin].copy()

    # Drop rows with missing values
    coin_df = coin_df.dropna(subset=features + [target])

    if len(coin_df) < 3:  # Skip if not enough data
        print(f"⚠ Not enough data for {coin}, skipping...")
        continue

    # Define X (independent variables) and y (target variable)
    X = coin_df[features]
    y = coin_df[target]

    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n🔹 {coin} Model Performance:")
    print(f"   - Mean Absolute Error (MAE): {mae}")
    print(f"   - R² Score: {r2}")

    # Predict next price using the latest available data
    future_price = model.predict(X.iloc[-1:].values)[0]
    
    # Store results
    forecast_results[coin] = future_price

# Convert results to DataFrame
forecast_df = pd.DataFrame(list(forecast_results.items()), columns=['Coin', 'Predicted Price'])

# Save predictions
forecast_df.to_csv("crypto_price_forecast_linear.csv", index=False)
print("\n✅ Forecasting Complete! Predictions saved in 'crypto_price_forecast_linear.csv'.")
print(forecast_df)


import pandas as pd

# Load cleaned dataset
file_path = "cleaned_crypto_prices.csv"
df = pd.read_csv(file_path)

# Select the top 10 cryptocurrencies
top_10_coins = ["Bitcoin", "Ethereum", "Tether USDt", "BNB", "XRP", "USD Coin",
                "Dora Factory (new)", "Lido Staked ETH", "Cardano"]

# Count records for each coin
coin_counts = df[df['name'].isin(top_10_coins)]['name'].value_counts()
print("\n🔹 Available Records for Each Coin:")
print(coin_counts)

