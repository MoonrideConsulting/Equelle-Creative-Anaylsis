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

def main_dashboard():
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
        
        # Initialize the date range to None to avoid errors
        date_range = st.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            key='date_range'
        )

        # Ensure both start_date and end_date are selected
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date  # Default to full date range if not fully selected

    # Filter the data based on Batch and Date before creating the combination table
    filtered_data = filter_data(data, selected_batch, start_date, end_date)

    # Define available columns for selection
    available_columns = ['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type']

    # Let the user select which variables to include in the analysis, with default set to Messaging and Creative Theme
    selected_columns = st.multiselect(
        'Select Variables to Include in Analysis:',
        available_columns,  # Full list of options
        default=['Messaging Theme', 'Creative Theme']  # Default selection
    )

    # Control for the number of combinations
    num_combos = len(selected_columns)

    # Cross Sectional Analysis
    # Generate the combo table
    combo_table = cross_section_analysis(filtered_data, num_combos, selected_columns)

    # Step 5: Add Min/Max input boxes for Spend filtering
    st.write("Filter by Spend")
    spend_min = combo_table['Amount Spent'].min()
    spend_max = combo_table['Amount Spent'].max()

    # Create two input boxes for min and max spend with default values set to the min/max of the table
    min_spend = st.number_input("Min Spend", min_value=0, value=int(spend_min))
    max_spend = st.number_input("Max Spend", min_value=0, value=int(spend_max))

    # Apply the Spend Filter to the combo table (after the table is generated)
    combo_table = combo_table[(combo_table['Amount Spent'] >= min_spend) & (combo_table['Amount Spent'] <= max_spend)]

    # Display the filtered combo table
    st.dataframe(combo_table, use_container_width=True)

password_protection()
