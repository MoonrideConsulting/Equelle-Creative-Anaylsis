import streamlit as st
import pandas as pd
import itertools
from google.oauth2 import service_account
from google.cloud import bigquery
from datetime import datetime, timedelta
import re

def password_protection():
        main_dashboard()

# Function to extract the Batch information from Ad Name
def extract_batch(ad_name):
    match = re.search(r'Batch.*', ad_name)
    return match.group(0) if match else 'No Batch'

# Function to filter data based on selected criteria
def filter_data(data, selected_batch, start_date, end_date, selected_filters):
    # Apply Batch filter
    if selected_batch != "All":
        data = data[data['Batch'] == selected_batch]
    
    # Apply Date filter only if both start and end dates are selected
    if start_date and end_date:
        data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    # Apply filters for selected variables
    for column, selected_value in selected_filters.items():
        if selected_value != "All":
            data = data[data[column] == selected_value]

    return data

# Function to create the cross-sectional analysis table
def cross_section_analysis(data, num_combos, selected_columns):
    # Generate all combinations of the specified number of columns
    combinations = list(itertools.combinations(selected_columns, num_combos))

    # Create an empty dataframe to store all results
    combined_results = pd.DataFrame()

    # Loop through each combination of columns and filter rows that match the combination
    for combo in combinations:
        # Group by the combination of columns and aggregate required metrics
        grouped = data.groupby(list(combo)).agg({
            'Spend': 'sum',
            'Clicks': 'sum',
            'Impressions': 'sum',
            'Purchases': 'sum'
        }).reset_index()

        # Calculate additional metrics
        grouped['CPM'] = round((grouped['Spend'] / grouped['Impressions']) * 1000, 2)
        grouped['CPA'] = round(grouped['Spend'] / grouped['Purchases'], 2)
        grouped['CPC'] = round(grouped['Spend'] / grouped['Clicks'], 2)
        grouped['Spend'] = round(grouped['Spend'], 0)

        # Combine the values in the columns to create a 'Combination' identifier
        grouped['Combination'] = grouped.apply(lambda row: ', '.join([f"{col}={row[col]}" for col in combo]), axis=1)

        # Append the results to the combined dataframe
        combined_results = pd.concat([combined_results, grouped[['Combination', 'Purchases', 'Spend', 'Clicks', 'Impressions', 'CPM', 'CPA', 'CPC']]])

    # Sort the results by Purchases in descending order
    combined_results = combined_results.sort_values(by='Purchases', ascending=False)

    return combined_results

# Main dashboard function
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
        WHERE DATE(Date) >= "2024-01-01";"""
        st.session_state.full_data = pd.read_gbq(query, credentials=credentials)

    # Data preparation
    data = st.session_state.full_data
    data.columns = data.columns.str.replace('__Facebook_Ads', '', regex=False)
    data.columns = data.columns.str.replace('_', ' ', regex=False)

    # Change Names of Clicks and Spend columns
    data.rename(columns={'Clicks all': 'Clicks', 'Amount Spent': 'Spend'}, inplace=True)

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

    # Define available columns for selection including Creative Imagery and Text Hook
    available_columns = ['Ad Format', 'Creative Theme', 'Messaging Theme', 'Landing Page Type', 'Creative Imagery', 'Text Hook']

    # Let the user select which variables to include in the analysis, with default set to Messaging and Creative Theme
    selected_columns = st.multiselect(
        'Select Variables to Include in Analysis:',
        available_columns,  # Full list of options
        default=['Messaging Theme', 'Creative Theme']  # Default selection
    )

    # Step 4: Add filters for each selected column
    selected_filters = {}
    for column in selected_columns:
        unique_values = ["All"] + sorted(data[column].dropna().unique())
        selected_value = st.selectbox(f'Filter by {column}:', unique_values, index=0)
        selected_filters[column] = selected_value

    # Control for the number of combinations
    num_combos = len(selected_columns)

    # Step 5: Filter the data based on Batch, Date, and selected column filters
    filtered_data = filter_data(data, selected_batch, start_date, end_date, selected_filters)

    # Step 6: Generate the cross-sectional analysis table
    combo_table = cross_section_analysis(filtered_data, num_combos, selected_columns)

    display_df = combo_table.copy()
    display_df['Spend'] = display_df['Spend'].apply(lambda x: f"${x:,.0f}")
    display_df['CPM'] = display_df['CPM'].apply(lambda x: f"${x:,.2f}")
    display_df['CPA'] = display_df['CPA'].apply(lambda x: f"${x:,.2f}")
    display_df['CPC'] = display_df['CPC'].apply(lambda x: f"${x:,.2f}")

    # Step 7: Display the filtered combo table
    st.dataframe(display_df, use_container_width=True)

# Run the dashboard
password_protection()
