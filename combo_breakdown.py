import streamlit as st
import pandas as pd

# Function to filter the data based on selected criteria
def filter_data(data, selected_batch, start_date, end_date, selected_messaging_theme, selected_creative_theme, selected_format, selected_landing_page, selected_creative_imagery, selected_text_hook):
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

    # Apply Creative Imagery filter
    if selected_creative_imagery != "All":
        data = data[data['Creative Imagery'] == selected_creative_imagery]

    # Apply Text Hook filter
    if selected_text_hook != "All":
        data = data[data['Text Hook'] == selected_text_hook]

    return data

# Main function for Combo Breakdown page
def main():
    # Load the data from session state (assuming it's already loaded)
    data = st.session_state.full_data

    # Prepare the data (assuming relevant columns exist)
    data.columns = data.columns.str.replace('__Facebook_Ads', '', regex=False)
    data.columns = data.columns.str.replace('_', ' ', regex=False)

    # Change Names of Clicks and Spend columns
    data.rename(columns={'Clicks all': 'Clicks', 'Amount Spent': 'Spend'}, inplace=True)

    # Fix missing batches
    data['Batch'].fillna('No Batch', inplace=True)

    # Step 1: Create a new "Batch" column from "Ad Name" (if it doesn't exist already)
    if 'Batch' not in data.columns:
        data['Batch'] = data['Ad Name'].apply(lambda x: x.split('Batch')[-1].strip() if 'Batch' in x else 'No Batch')
    
    # Create a copy of the original data for the combo breakdown page
    data_copy = data.copy()

    # Replace None or missing values with 'N/A' in critical columns in the copied dataset
    data_copy['Messaging Theme'].fillna('N/A', inplace=True)
    data_copy['Creative Theme'].fillna('N/A', inplace=True)
    data_copy['Ad Format'].fillna('N/A', inplace=True)
    data_copy['Landing Page Type'].fillna('N/A', inplace=True)
    data_copy['Creative Imagery'].fillna('N/A', inplace=True)
    data_copy['Text Hook'].fillna('N/A', inplace=True)

    # Step 2: Batch and Date filters at the top
    col1, col2 = st.columns(2)

    with col1:
        # Batch filter
        batch_options = ["All"] + sorted(data_copy['Batch'].unique())
        selected_batch = st.selectbox('Select Batch:', batch_options, index=0)

    with col2:
        # Date range filter
        min_date = data_copy['Date'].min()
        max_date = data_copy['Date'].max()
        date_range = st.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

        # Ensure both start_date and end_date are selected
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date  # Default to full date range if not fully selected

    # Step 3: Additional filters for Messaging Theme, Creative Theme, Ad Format, Landing Page Type, Creative Imagery, and Text Hook
    col3, col4, col5, col6, col7, col8 = st.columns(6)

    with col3:
        # Messaging Theme filter
        messaging_theme_options = ["All"] + sorted(data_copy['Messaging Theme'].unique())
        selected_messaging_theme = st.selectbox('Select Messaging Theme:', messaging_theme_options, index=0)

    with col4:
        # Creative Theme filter
        creative_theme_options = ["All"] + sorted(data_copy['Creative Theme'].unique())
        selected_creative_theme = st.selectbox('Select Creative Theme:', creative_theme_options, index=0)

    with col5:
        # Ad Format filter
        format_options = ["All"] + sorted(data_copy['Ad Format'].unique())
        selected_format = st.selectbox('Select Ad Format:', format_options, index=0)

    with col6:
        # Landing Page Type filter
        landing_page_options = ["All"] + sorted(data_copy['Landing Page Type'].unique())
        selected_landing_page = st.selectbox('Select Landing Page Type:', landing_page_options, index=0)

    with col7:
        # Creative Imagery filter
        creative_imagery_options = ["All"] + sorted(data_copy['Creative Imagery'].unique())
        selected_creative_imagery = st.selectbox('Select Creative Imagery:', creative_imagery_options, index=0)

    with col8:
        # Text Hook filter
        text_hook_options = ["All"] + sorted(data_copy['Text Hook'].unique())
        selected_text_hook = st.selectbox('Select Text Hook:', text_hook_options, index=0)

    # Step 4: Filter the data based on selected criteria
    filtered_data = filter_data(
        data_copy, selected_batch, start_date, end_date, 
        selected_messaging_theme, selected_creative_theme, 
        selected_format, selected_landing_page, 
        selected_creative_imagery, selected_text_hook
    )

    # Step 5: Group the data by Ad Name and aggregate the numeric columns
    grouped_data = filtered_data.groupby('Ad Name').agg({
        'Messaging Theme': 'first',
        'Creative Theme': 'first',
        'Ad Format': 'first',
        'Landing Page Type': 'first',
        'Creative Imagery': 'first',
        'Text Hook': 'first',
        'Purchases': 'sum',
        'Spend': 'sum',
        'Clicks': 'sum',
        'Impressions': 'sum',
        'Ad Preview Shareable Link': 'first'  # Get the first Ad Preview Link
    }).reset_index()

    # Calculate additional metrics
    grouped_data['CPM'] = round((grouped_data['Spend'] / grouped_data['Impressions']) * 1000, 2)
    grouped_data['CPA'] = round(grouped_data['Spend'] / grouped_data['Purchases'], 2)
    grouped_data['CPC'] = round(grouped_data['Spend'] / grouped_data['Clicks'], 2)
    grouped_data['Spend'] = round(grouped_data['Spend'], 0)

    # Step 6: Add the 'Ad Preview Shareable Link' as a clickable link behind 'Ad Name'
    grouped_data['Ad Name'] = grouped_data.apply(
        lambda row: f'<a href="{row["Ad Preview Shareable Link"]}" target="_blank">{row["Ad Name"]}</a>',
        axis=1
    )

    # Step 7: Drop the 'Ad Preview Shareable Link' column from display as we already linked it
    grouped_data = grouped_data.drop(columns=['Ad Preview Shareable Link'])
    grouped_data = grouped_data.sort_values(by='Spend', ascending=False)
    
    # Format the monetary columns for display
    display_df = grouped_data.copy()
    display_df['Spend'] = display_df['Spend'].apply(lambda x: f"${x:,.0f}")
    display_df['CPM'] = display_df['CPM'].apply(lambda x: f"${x:,.2f}")
    display_df['CPA'] = display_df['CPA'].apply(lambda x: f"${x:,.2f}")
    display_df['CPC'] = display_df['CPC'].apply(lambda x: f"${x:,.2f}")

    # Step 8: Display the dataframe with clickable links using st.markdown and HTML
    st.markdown(display_df.to_html(escape=False), unsafe_allow_html=True)
