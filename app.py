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

#For random forest modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import numpy as np

#plotting
import matplotlib.pyplot as plt
import altair as alt

st.set_page_config(page_title="Equelle Creative Analysis",page_icon="🧑‍🚀",layout="wide")

def password_protection():
        main_dashboard()

def cross_section_analysis(data, num_combos):
    # Columns to be used for combinations
    columns = ['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type']

    # Generate all combinations of the specified number of columns
    combinations = list(itertools.combinations(columns, num_combos))

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

def linear_regression_analysis(data, var):
    # Select relevant columns
    features = ['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type']
    target = var

    # Separate the input features (X) and target variable (y)
    X = data[features]
    y = data[target]

    # One-Hot Encoding for categorical features
    X_encoded = pd.get_dummies(X[['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type']], drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Initialize and train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Get the model coefficients
    coefficients = model.coef_

    # Create a DataFrame for feature importance (based on coefficients)
    feature_importance_df = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Coefficient': coefficients
    })

    # Sort the features by absolute value of coefficients
    feature_importance_df = feature_importance_df.sort_values(by='Coefficient', key=abs, ascending=False)

    return feature_importance_df

def plot_linear_regression_coefficients(feature_importance_df):
    # Sort the dataframe by absolute value of coefficients
    feature_importance_df = feature_importance_df.sort_values(by='Coefficient', key=abs, ascending=False)

    # Create an Altair horizontal bar chart for coefficients
    chart = alt.Chart(feature_importance_df).mark_bar().encode(
        x=alt.X('Coefficient', title='Coefficient'),
        y=alt.Y('Feature', sort='-x', title = '', axis=alt.Axis(labelLimit=400)),  # Adjust label width
        color=alt.Color('Coefficient', scale=alt.Scale(scheme='blueorange'))
    ).properties(
        title='Feature Importance (Linear Regression Coefficients)',
        width=600  # Set the width of the chart
    )

    # Display the chart in Streamlit
    st.altair_chart(chart, use_container_width=True)


def main_dashboard():
    st.markdown("<h1 style='text-align: center;'>Equelle Creative Analysis</h1>", unsafe_allow_html=True)
    # Calculate the date one year ago from today
    one_year_ago = (datetime.now() - timedelta(days=365)).date()
    
    if 'full_data' not in st.session_state:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(credentials=credentials)
        # Modify the query
        query = f"""
        SELECT *
        FROM `Equelle_Segments.equelle_ad_level_all`
        WHERE DATE(Date) > "2024-01-01";"""
        st.session_state.full_data = pandas.read_gbq(query, credentials=credentials)

    # Rename Cols / Clean Up Df
    data = st.session_state.full_data
    data.columns = data.columns.str.replace('__Facebook_Ads', '', regex=False)
    data.columns = data.columns.str.replace('_', ' ', regex=False)

    num_combos = st.number_input("Change number of combinations", 2, 4)

    st.header("Cross Sectional Analysis")
    st.write("This chart allows you to see metrics across combinations of the 4 variables of interest (Ad Format, Messaging Theme, Creative Theme, and Landing Page Type). By default it is sorted by the number of
             purchases by can be sorted by other columns by clicking on them (once for desc, twice for asc). You can change the number of variables in the combination by editing the number selecter below.")
    st.dataframe(cross_section_analysis(data, num_combos), use_container_width=True)

    st.header("ML Analysis")
    cleaned_data = data.dropna()
    cleaned_data = cleaned_data.loc[cleaned_data['Messaging Theme'] != 'N/A']
    model_data = prep_data(cleaned_data)

    # Create a select box to choose the metric
    metric = st.selectbox('Select a Metric', ['Purchases', 'Clicks all', 'Amount Spent'])
        
    col1, col2 =  st.columns(2)    
    with col1:       
        #random forest analysis
        feature_importance_df = feature_importance_analysis(model_data, metric)
        streamlit_feature_importance_bar_chart(feature_importance_df)

    with col2: 
        #regression analysis
        regression_data = linear_regression_analysis(model_data, metric) 
        plot_linear_regression_coefficients(regression_data)

password_protection()
