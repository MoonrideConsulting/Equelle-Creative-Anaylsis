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

st.set_page_config(page_title="Equelle Creative Analysis",page_icon="🧑‍🚀",layout="wide")

def password_protection():
        main_dashboard()

# Function to extract the Batch information from Ad Name
def extract_batch(ad_name):
    match = re.search(r'Batch.*', ad_name)
    return match.group(0) if match else 'No Batch'

# Function to generate interaction terms
def generate_interaction_terms(X_encoded, level):
    if level == 1:  # No interaction terms
        return X_encoded  # This case should only be used if you want individual features (we'll focus on combinations)
    else:
        # Create polynomial features for interaction terms
        poly = PolynomialFeatures(degree=level, interaction_only=True, include_bias=False)
        X_interactions = poly.fit_transform(X_encoded)
        
        # Get feature names for the interaction terms
        interaction_feature_names = poly.get_feature_names_out(X_encoded.columns)
        
        # Return the DataFrame with interaction terms (no individual features)
        return pd.DataFrame(X_interactions, columns=interaction_feature_names)

# Function to filter data before creating the combo table
def filter_data(data, selected_batch, start_date, end_date):
    # Apply the Batch filter
    if selected_batch != "All":
        data = data[data['Batch'] == selected_batch]
    
    # Apply the Date filter only if both start and end dates are selected
    if start_date and end_date:
        data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    
    return data

def cross_section_analysis(data, num_combos, selected_columns):
    # Generate all combinations of the specified number of columns from user-selected columns
    combinations = list(itertools.combinations(selected_columns, num_combos))

    # Create an empty dataframe to store all results
    combined_results = pd.DataFrame()

    # Loop through each combination of columns and filter rows that match the combination
    for combo in combinations:
        # Group by the combination of columns and aggregate required metrics
        grouped = data.groupby(list(combo)).agg({
            'Amount Spent': 'sum',
            'Clicks all': 'sum',
            'Impressions': 'sum',
            'Purchases': 'sum'
        }).reset_index()

        # Calculate additional metrics
        grouped['CPM'] = round((grouped['Amount Spent'] / grouped['Impressions']) * 1000, 2)
        grouped['CPA'] = round(grouped['Amount Spent'] / grouped['Purchases'], 2)
        grouped['CPC'] = round(grouped['Amount Spent'] / grouped['Clicks all'], 2)
        grouped['Amount Spent'] = round(grouped['Amount Spent'], 0)

        # Combine the values in the columns to create a 'Combination' identifier
        grouped['Combination'] = grouped.apply(lambda row: ', '.join([f"{col}={row[col]}" for col in combo]), axis=1)

        # Append the results to the combined dataframe
        combined_results = pd.concat([combined_results, grouped[['Combination', 'Purchases', 'Amount Spent', 'Clicks all', 'Impressions', 'CPM', 'CPA', 'CPC']]])

    # Sort the results by Purchases in descending order
    combined_results = combined_results.sort_values(by='Purchases', ascending=False)

    return combined_results

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

# Function to generate interaction terms
def generate_interaction_terms(X_encoded, level):
    if level == 1:  # No interaction terms
        return X_encoded  # This case should only be used if you want individual features (we'll focus on combinations)
    else:
        # Create polynomial features for interaction terms
        poly = PolynomialFeatures(degree=level, interaction_only=True, include_bias=False)
        X_interactions = poly.fit_transform(X_encoded)
        
        # Get feature names for the interaction terms
        interaction_feature_names = poly.get_feature_names_out(X_encoded.columns)
        
        # Return the DataFrame with interaction terms (no individual features)
        return pd.DataFrame(X_interactions, columns=interaction_feature_names)

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

def main_dashboard():
    st.markdown("<h1 style='text-align: center;'>Equelle Creative Analysis</h1>", unsafe_allow_html=True)

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

    # Step 1: Create a new "Batch" column from "Ad Name"
    data['Batch'] = data['Ad Name'].apply(extract_batch)

    # Step 2: Create columns for side-by-side layout
    col1, col2 = st.columns(2)

    # Step 3: Batch and Date filters inside columns
    with col1:
        # Add a Batch filter
        batch_options = ["All"] + sorted(data['Batch'].unique())
        selected_batch = st.selectbox('Select Batch:', batch_options, index=0)

    with col2:
        # Add a Date range filter
        min_date = data['Date'].min()
        max_date = data['Date'].max()
        start_date, end_date = st.date_input(
            "Select Date Range",
            [None, None],
            min_value=min_date,
            max_value=max_date,
            key='date_range'
        )

    # Filter the data based on Batch and Date before creating the combination table
    filtered_data = filter_data(data, selected_batch, start_date, end_date)

    # Define available columns for selection
    available_columns = ['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type']

    # Let the user select which variables to include in the analysis
    selected_columns = st.multiselect('Select Variables to Include in Analysis:', available_columns, default=available_columns)

    # Control for the number of combinations
    num_combos = len(selected_columns)

    # Cross Sectional Analysis
    st.header("Cross Sectional Analysis")
    st.write("This chart allows you to see metrics across combinations of the selected variables. By default, it is sorted by the number of purchases but can be sorted by other columns by clicking on them. Adjust the number of variables in the combination by changing the selector below.")
    st.dataframe(cross_section_analysis(filtered_data, num_combos, selected_columns), use_container_width=True)

    # ML Analysis Section (we can leave this for now, but adding filter flexibility)
    st.header("ML Analysis")
    st.write("This chart shows the output of a regression model looking at how combinations of the selected variables influenced the chosen metric.")
    cleaned_data = data.dropna()
    cleaned_data = cleaned_data.loc[cleaned_data['Messaging Theme'] != 'N/A']
    model_data = prep_data(cleaned_data)
        
    #col1, col2 =  st.columns(2)    
    #with col1:       
        #random forest analysis
        #feature_importance_df = feature_importance_analysis(model_data, metric)
        #streamlit_feature_importance_bar_chart(feature_importance_df)

    #with col2: 

    # Run the linear regression analysis based on user selection and selected metric (e.g., Purchases)
    selected_metric = st.selectbox('Select a Metric', ['Purchases', 'Clicks', 'Spend'])

    # Perform linear regression with interaction terms
    feature_importance_df = linear_regression_analysis(model_data, selected_metric, num_combos)

    # Plot the resulting feature importance
    plot_linear_regression_coefficients(feature_importance_df)

password_protection()
