import streamlit as st
import pandas as pd

# Function to rank each value of a variable based on Purchases
def rank_by_purchases(data, column):
    ranking = data.groupby(column).agg({
        'Purchases': 'sum'
    }).reset_index().sort_values(by='Purchases', ascending=False)

    return ranking

# Function to get combination rankings for each value of the variable, with CPA, CPC, and CPM
def rank_combinations(data, main_column, secondary_column):
    # Group by the combination of main_column and secondary_column
    combo_rankings = data.groupby([main_column, secondary_column]).agg({
        'Purchases': 'sum',
        'Amount Spent': 'sum',
        'Clicks all': 'sum',
        'Impressions': 'sum'
    }).reset_index()

    # Calculate additional metrics
    combo_rankings['CPA'] = combo_rankings['Amount Spent'] / combo_rankings['Purchases']  # Cost per Acquisition
    combo_rankings['CPC'] = combo_rankings['Amount Spent'] / combo_rankings['Clicks all']  # Cost per Click
    combo_rankings['CPM'] = (combo_rankings['Amount Spent'] / combo_rankings['Impressions']) * 1000  # Cost per 1000 Impressions

    # Sort the table by Purchases
    combo_rankings = combo_rankings.sort_values(by='Purchases', ascending=False)

    return combo_rankings

# Function to filter data based on Batch and Date
def filter_data(data, selected_batch, start_date, end_date):
    # Apply the Batch filter
    if selected_batch != "All":
        data = data[data['Batch'] == selected_batch]
    
    # Apply the Date filter only if both start_date and end_date are selected
    if start_date and end_date:
        data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    
    return data

# Main function to display ranked combos with filters
def main():
    # Load the data from session state (assuming it's already loaded)
    data = st.session_state.full_data
    
    # Prepare the data (assuming relevant columns like 'Messaging Theme' and 'Creative Theme' exist)
    data.columns = data.columns.str.replace('__Facebook_Ads', '', regex=False)
    data.columns = data.columns.str.replace('_', ' ', regex=False)

    # Step 1: Create a new "Batch" column from "Ad Name" (if it doesn't exist already)
    if 'Batch' not in data.columns:
        data['Batch'] = data['Ad Name'].apply(lambda x: x.split('Batch')[-1].strip() if 'Batch' in x else 'No Batch')

    # Step 2: Create filters for Batch and Date
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

    # Step 3: Filter the data based on Batch and Date
    filtered_data = filter_data(data, selected_batch, start_date, end_date)

    # Step 4: Rank Messaging Theme by Purchases
    messaging_theme_ranking = rank_by_purchases(filtered_data, 'Messaging Theme')
    creative_theme_ranking = rank_by_purchases(filtered_data, 'Creative Theme')

    # Step 5: Display rankings for Messaging Theme
    st.subheader("Messaging Theme Rankings (by Purchases)")
    for _, row in messaging_theme_ranking.iterrows():
        theme_value = row['Messaging Theme']
        st.write(f"**{theme_value}** - Purchases: {row['Purchases']}")
        
        # Get combination rankings with Creative Theme
        combo_rankings = rank_combinations(filtered_data, 'Messaging Theme', 'Creative Theme')
        filtered_combos = combo_rankings[combo_rankings['Messaging Theme'] == theme_value]

        # Display combinations DataFrame in the dropdown
        with st.expander(f"See combinations with Creative Theme for {theme_value}"):
            st.dataframe(filtered_combos)

    # Step 6: Display rankings for Creative Theme
    st.subheader("Creative Theme Rankings (by Purchases)")
    for _, row in creative_theme_ranking.iterrows():
        theme_value = row['Creative Theme']
        st.write(f"**{theme_value}** - Purchases: {row['Purchases']}")
        
        # Get combination rankings with Messaging Theme
        combo_rankings = rank_combinations(filtered_data, 'Creative Theme', 'Messaging Theme')
        filtered_combos = combo_rankings[combo_rankings['Creative Theme'] == theme_value]

        # Display combinations DataFrame in the dropdown
        with st.expander(f"See combinations with Messaging Theme for {theme_value}"):
            st.dataframe(filtered_combos)
