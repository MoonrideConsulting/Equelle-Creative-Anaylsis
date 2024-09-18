import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandas_gbq
import pandas 
import itertools
from google.oauth2 import service_account
from google.cloud import bigquery
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import re

#For random forest modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import numpy as np

#plotting
import matplotlib.pyplot as plt
import altair as alt

def prep_data(data):
    #Remove NAs
    features = ['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type', 'Amount Spent', 'Clicks all', 'Impressions']
    data[features] = data[features].replace("", np.nan)
    cleaned_data = data.dropna()
    cleaned_data = cleaned_data.loc[cleaned_data['Messaging Theme'] != 'N/A']

    #Group data
    model_data = cleaned_data.groupby(['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type']).agg({
        'Amount Spent': 'sum',               # Sum 'Spend'
        'Clicks all': 'sum',              # Sum 'Clicks'
        'Impressions': 'sum',         # Sum 'Impressions'
        'Purchases': 'sum'            # Sum 'Purchases'
    }).reset_index()

    model_data['CPM'] = round((model_data['Amount Spent'] / model_data['Impressions']) * 1000, 2)
    model_data['CPA'] = round(model_data['Amount Spent'] / model_data['Purchases'], 2)
    model_data['CPC'] = round(model_data['Amount Spent'] / model_data['Clicks all'], 2)
    model_data['Amount Spent'] = round(model_data['Amount Spent'], 0)
    model_data.dropna(inplace = True)
        
    return model_data

# Function to prepare data and train a Random Forest model
def feature_importance_analysis(data, var):
    # Select relevant columns
    features = ['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type', 'Amount Spent', 'Clicks all', 'Impressions']
    target = var

    # Separate the input features (X) and target variable (y)
    X = data[features]
    y = data[target]

    # One-Hot Encoding for categorical features
    X_encoded = pd.get_dummies(X[['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type']], drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Initialize and train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Get feature importances
    feature_importances = model.feature_importances_

    # Create a DataFrame for feature importance ranking
    feature_importance_df = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Importance': feature_importances
    })

    # Sort the features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df

def streamlit_feature_importance_bar_chart(feature_importance_df):
    # Sort the dataframe by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Create an Altair horizontal bar chart with wider y-axis
    chart = alt.Chart(feature_importance_df).mark_bar().encode(
        x=alt.X('Importance', title='Importance'),
        y=alt.Y('Feature', sort='-x', title = '', axis=alt.Axis(labelLimit=400)),  # Adjust label width
        color=alt.Color('Importance', scale=alt.Scale(scheme='blues'))
    ).properties(
        title='Feature Importance',
        width=600  # Adjust the chart width if needed
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

# Main linear regression analysis function with handling of empty strings and combinations
def linear_regression_analysis(data, var, combination_level):
    # Select relevant columns
    features = ['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type']
    target = var

    # Separate the input features (X) and target variable (y)
    X = data[features]
    y = data[target]

    # One-Hot Encoding for categorical features
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Generate interaction terms based on the user-selected combination level
    X_interactions = generate_interaction_terms(X_encoded, combination_level)

    # Remove individual features from the interaction terms
    if combination_level > 1:
        # Filter only interaction terms (exclude individual feature terms)
        feature_columns = [col for col in X_interactions.columns if "_" in col]
        X_interactions = X_interactions[feature_columns]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_interactions, y, test_size=0.2, random_state=42)

    # Initialize and train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get the model coefficients
    coefficients = model.coef_

    # Create a DataFrame for feature importance (based on coefficients)
    feature_importance_df = pd.DataFrame({
        'Feature': X_interactions.columns,
        'Coefficient': coefficients
    })

    # Sort the features by absolute value of coefficients
    feature_importance_df = feature_importance_df.sort_values(by='Coefficient', key=abs, ascending=False)

    return feature_importance_df

# Plotting function remains the same with the hidden legend
def plot_linear_regression_coefficients(feature_importance_df):
    # Sort the dataframe by absolute value of coefficients
    feature_importance_df = feature_importance_df.sort_values(by='Coefficient', key=abs, ascending=False)

    # Create an Altair horizontal bar chart for coefficients
    chart = alt.Chart(feature_importance_df).mark_bar().encode(
        x=alt.X('Coefficient', title='Coefficient'),
        y=alt.Y('Feature', sort='-x', title='', axis=alt.Axis(labelLimit=400)),  # Adjust label width
        color=alt.Color('Coefficient', scale=alt.Scale(scheme='blueorange'), legend=None)  # Hide the legend
    ).properties(
        title='Feature Importance (Linear Regression Coefficients)',
        width=600  # Set the width of the chart
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

def main():
    # Load data if not already in session state
    if 'full_data' not in st.session_state:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(credentials=credentials)
        query = f"""
        SELECT *
        FROM `Equelle_Segments.equelle_ad_level_all`
        WHERE DATE(Date) > "2024-01-01";"""
        st.session_state.full_data = pandas.read_gbq(query, credentials=credentials)

    # Data preparation
    data = st.session_state.full_data
    data.columns = data.columns.str.replace('__Facebook_Ads', '', regex=False)
    data.columns = data.columns.str.replace('_', ' ', regex=False)

    # ML Analysis Section (we can leave this for now, but adding filter flexibility)
    cleaned_data = data.dropna()
    cleaned_data = cleaned_data.loc[cleaned_data['Messaging Theme'] != 'N/A']
    model_data = prep_data(cleaned_data)
        
    col1, col2 =  st.columns(2)    
    with col1:       
        #random forest analysis
        feature_importance_df = feature_importance_analysis(model_data, metric)
        streamlit_feature_importance_bar_chart(feature_importance_df)

    with col2: 
        # Run the linear regression analysis based on user selection and selected metric (e.g., Purchases)
        selected_metric = st.selectbox('Select a Metric', ['Purchases', 'Clicks', 'Spend'])

        # Perform linear regression with interaction terms
        feature_importance_df = linear_regression_analysis(model_data, selected_metric, num_combos)

        # Plot the resulting feature importance
        plot_linear_regression_coefficients(feature_importance_df)

