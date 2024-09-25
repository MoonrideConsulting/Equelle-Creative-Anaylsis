import streamlit as st
import pandas as pd

# Function to filter the data based on selected criteria
def filter_data(data, selected_batch, start_date, end_date, selected_messaging_theme, selected_creative_theme, selected_format, selected_landing_page):
    # Apply Batch and Date filters
    if selected_batch != "All":
        data = data[data['Batch'] == selected_batch]

    if start_date and end_date:
        data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    # Apply Messaging Theme filter
    if selected_messaging_theme != "All":
        data = data[data['Messaging Theme'] == selected_messaging_theme]

    # Apply Creative Theme filter
    if selected_creative_theme != "All":
        data = data[data['Creative Theme'] == selected_creative_theme]

    # Apply Ad Format filter
    if selected_format != "All":
        data = data[data['Ad Format'] == selected_format]

    # Apply Landing Page Type filter
    if selected_landing_page != "All":
        data = data[data['Landing Page Type'] == selected_landing_page]

    return data

# Main function for Combo Breakdown page
def main():
    # Load the data from session state (assuming it's already loaded)
    data = st.session_state.full_data

    # Prepare the data (assuming relevant columns like 'Messaging Theme' and 'Creative Theme' exist)
    data.columns = data.columns.str.replace('__Facebook_Ads', '', regex=False)
    data.columns = data.columns.str.replace('_', ' ', regex=False)

    # Replace None or missing values with 'N/A' in critical columns
    data['Messaging Theme'].fillna('N/A', inplace=True)
    data['Creative Theme'].fillna('N/A', inplace=True)
    data['Batch'].fillna('No Batch', inplace=True)
    data['Ad Format'].fillna('N/A', inplace=True)
    data['Landing Page Type'].fillna('N/A', inplace=True)

    # Step 1: Create a new "Batch" column from "Ad Name" (if it doesn't exist already)
    if 'Batch' not in data.columns:
        data['Batch'] = data['Ad Name'].apply(lambda x: x.split('Batch')[-1].strip() if 'Batch' in x else 'No Batch')

    # Step 2: Batch and Date filters at the top
    col1, col2 = st.columns(2)

    with col1:
        # Batch filter
        batch_options = ["All"] + sorted(data['Batch'].unique())
        selected_batch = st.selectbox('Select Batch:', batch_options, index=0)

    with col2:
        # Date range filter
        min_date = data['Date'].min()
        max_date = data['Date'].max()
        date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

        # Ensure both start_date and end_date are selected
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date  # Default to full date range if not fully selected

    # Step 3: Additional filters for Messaging Theme, Creative Theme, Ad Format, and Landing Page Type
    col3, col4, col5, col6 = st.columns(4)

    with col3:
        # Messaging Theme filter
        messaging_theme_options = ["All"] + sorted(data['Messaging Theme'].unique())
        selected_messaging_theme = st.selectbox('Select Messaging Theme:', messaging_theme_options, index=0)

    with col4:
        # Creative Theme filter
        creative_theme_options = ["All"] + sorted(data['Creative Theme'].unique())
        selected_creative_theme = st.selectbox('Select Creative Theme:', creative_theme_options, index=0)

    with col5:
        # Ad Format filter
        format_options = ["All"] + sorted(data['Ad Format'].unique())
        selected_format = st.selectbox('Select Ad Format:', format_options, index=0)

    with col6:
        # Landing Page Type filter
        landing_page_options = ["All"] + sorted(data['Landing Page Type'].unique())
        selected_landing_page = st.selectbox('Select Landing Page Type:', landing_page_options, index=0)

    # Step 4: Filter the data based on selected criteria
    filtered_data = filter_data(data, selected_batch, start_date, end_date, selected_messaging_theme, selected_creative_theme, selected_format, selected_landing_page)

    # Step 5: Group the data by Ad Name and aggregate the numeric columns
    grouped_data = filtered_data.groupby('Ad Name').agg({
        'Messaging Theme': 'first',
        'Creative Theme': 'first',
        'Ad Format': 'first',
        'Landing Page Type': 'first',
        'Purchases': 'sum',
        'Amount Spent': 'sum',
        'Clicks all': 'sum',
        'Impressions': 'sum'
    }).reset_index()

    # Step 6: Calculate additional metrics
    grouped_data['CPA'] = grouped_data['Amount Spent'] / grouped_data['Purchases']  # Cost per Acquisition
    grouped_data['CPC'] = grouped_data['Amount Spent'] / grouped_data['Clicks all']  # Cost per Click
    grouped_data['CPM'] = (grouped_data['Amount Spent'] / grouped_data['Impressions']) * 1000  # Cost per 1000 Impressions

    # Step 7: Display the filtered and grouped data
    st.dataframe(grouped_data)

    st.dataframe(st.session_state.full_data)
