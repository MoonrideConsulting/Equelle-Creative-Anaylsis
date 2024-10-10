import streamlit as st
import pandas as pd
import plotly.express as px

# Function to rank each value of a variable based on Purchases
def rank_by_purchases(data, column):
    ranking = data.groupby(column).agg({
        'Purchases': 'sum',
        'Spend': 'sum',
        'Clicks': 'sum',
        'Impressions': 'sum'
    }).reset_index()

    # Calculate additional metrics
    ranking['CPM'] = (ranking['Spend'] / ranking['Impressions']) * 1000
    ranking['CPA'] = ranking['Spend'] / ranking['Purchases']
    ranking['CPC'] = ranking['Spend'] / ranking['Clicks']

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

# Function to create a treemap visualization with custom color rules
def create_treemap(data, main_column, secondary_column, theme_value):
    # Filter out rows where CPA is zero or NaN to avoid errors in the treemap
    data = data[data['CPA'] > 0]
    data = data[data['Purchases'] > 0]
    
    # Round CPA for better visual clarity
    data['CPA'] = round(data['CPA'], 2)

    # Define custom color map based on CPA value ranges
    def map_color(cpa_value):
        if cpa_value < 100:
            return 'green'
        elif 200 <= cpa_value < 250:
            return 'seagreen'
        elif 250 <= cpa_value < 300:
            return 'orange'
        else:
            return 'red'

    # Apply color mapping based on CPA values
    data['Color'] = data['CPA'].apply(map_color)
    
    # Create treemap with custom colors
    fig = px.treemap(
        data,
        path=[secondary_column],
        values='Purchases',
        color='Color',  # Using the mapped 'Color' column
        color_discrete_map={
            'green': 'green',
            'seagreen': 'seagreen',
            'orange': 'orange',
            'red': 'red'
        },
        title=f'Treemap of {theme_value} and {secondary_column}s',
        hover_data={
            'Purchases': True,
            'CPA': True,
        }
    )

    # Adjust the branch display and layout margins
    fig.update_traces(branchvalues='remainder')
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    
    return fig

# Main function to display ranked combos with filters
def main():
    # Load the data from session state (assuming it's already loaded)
    data = st.session_state.full_data
    
    # Prepare the data (assuming relevant columns like 'Messaging Theme' and 'Creative Theme' exist)
    data.columns = data.columns.str.replace('__Facebook_Ads', '', regex=False)
    data.columns = data.columns.str.replace('_', ' ', regex=False)

    # Change Names of Clicks and Spend columns
    data.rename(columns={'Clicks all': 'Clicks', 'Amount Spent': 'Spend'}, inplace=True)

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

    # Add Creative Imagery and Text Hook to available columns
    available_columns = ['Messaging Theme', 'Creative Theme', 'Ad Format', 'Landing Page Type', 'Creative Imagery', 'Text Hook']

    main_column = st.selectbox('Select Main Column:', available_columns, index=0)
    secondary_column = st.selectbox('Select Secondary Column:', available_columns, index=1)

    # Step 3: Filter the data based on Batch and Date
    filtered_data = filter_data(data, selected_batch, start_date, end_date)

    # Step 4: Rank Messaging Theme by Purchases and display as DataFrame
    ranking = rank_by_purchases(filtered_data, main_column)
    
    st.subheader(f"{main_column} Rankings (by Purchases)")
    
    # Show a DataFrame for each Messaging Theme
    for _, row in ranking.iterrows():
        theme_value = row[main_column]
        # Create a DataFrame for the current row
        df = pd.DataFrame([row], columns=[main_column, 'Purchases', 'Spend', 'Clicks', 'Impressions', 'CPM', 'CPA', 'CPC'])
        display_df = df.copy()
        display_df['Spend'] = display_df['Spend'].apply(lambda x: f"${x:,.0f}")
        display_df['CPM'] = display_df['CPM'].apply(lambda x: f"${x:,.2f}")
        display_df['CPA'] = display_df['CPA'].apply(lambda x: f"${x:,.2f}")
        display_df['CPC'] = display_df['CPC'].apply(lambda x: f"${x:,.2f}")
        st.write(display_df.reset_index(drop=True))

        # Get combination rankings with Creative Theme
        combo_rankings = rank_by_purchases(filtered_data, [main_column, secondary_column])
        filtered_combos = combo_rankings[combo_rankings[main_column] == theme_value]

        # Display treemap in the dropdown
        with st.expander(f"See combinations with {secondary_column} for {theme_value}"):
            # Create and display the treemap
            treemap_fig = create_treemap(filtered_combos, main_column, secondary_column, theme_value)
            st.plotly_chart(treemap_fig)

        # Divider between each ranking and its dropdown
        st.divider()
