import streamlit as st
import pandas as pd
import plotly.express as px

# Function to rank each value of a variable based on Purchases
def rank_by_purchases(data, column):
    ranking = data.groupby(column).agg({
        'Purchases': 'sum',
        'Amount Spent': 'sum',
        'Clicks all': 'sum',
        'Impressions': 'sum'
    }).reset_index()

    # Calculate additional metrics
    ranking['CPM'] = (ranking['Amount Spent'] / ranking['Impressions']) * 1000
    ranking['CPA'] = ranking['Amount Spent'] / ranking['Purchases']
    ranking['CPC'] = ranking['Amount Spent'] / ranking['Clicks all']

    # Sort by Purchases in descending order
    ranking = ranking.sort_values(by='Purchases', ascending=False)

    return ranking

# Function to filter data based on Batch and Date
def filter_data(data, selected_batch, start_date, end_date):
    # Apply the Batch filter
    if selected_batch != "All":
        data = data[data['Batch'] == selected_batch]

    # Apply the Date filter only if both start_date and end_date are selected
    if start_date and end_date:
        data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    return data

# Function to create a treemap visualization
def create_treemap(data, main_column, secondary_column):
    fig = px.treemap(
        data,
        path=[main_column, secondary_column],
        values='Purchases',
        color='CPA',
        color_continuous_scale='RdBu',
        title=f'Treemap of {main_column} and {secondary_column}',
        hover_data=['Purchases', 'CPA', 'Amount Spent', 'Clicks all', 'Impressions']
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    return fig

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

    # Step 4: Rank Messaging Theme by Purchases and display as DataFrame
    messaging_theme_ranking = rank_by_purchases(filtered_data, 'Messaging Theme')
    
    st.subheader("Messaging Theme Rankings (by Purchases)")
    
    # Show a DataFrame for each Messaging Theme
    for _, row in messaging_theme_ranking.iterrows():
        theme_value = row['Messaging Theme']
        # Create a DataFrame for the current row
        df = pd.DataFrame([row], columns=['Messaging Theme', 'Purchases', 'Amount Spent', 'Clicks all', 'Impressions', 'CPM', 'CPA', 'CPC'])
        st.dataframe(df)

        # Get combination rankings with Creative Theme
        combo_rankings = rank_by_purchases(filtered_data, 'Creative Theme')
        filtered_combos = combo_rankings[combo_rankings['Creative Theme'] == theme_value]

        # Display treemap in the dropdown
        with st.expander(f"See combinations with Creative Theme for {theme_value}"):
            # Create and display the treemap
            treemap_fig = create_treemap(filtered_combos, 'Messaging Theme', 'Creative Theme')
            st.plotly_chart(treemap_fig)

