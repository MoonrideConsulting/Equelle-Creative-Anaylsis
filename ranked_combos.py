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

# Main function to display ranked combos
def main():
    # Load the data from session state (assuming it's already loaded)
    data = st.session_state.full_data
    
    # Prepare the data (assuming relevant columns like 'Messaging Theme' and 'Creative Theme' exist)
    data.columns = data.columns.str.replace('__Facebook_Ads', '', regex=False)
    data.columns = data.columns.str.replace('_', ' ', regex=False)

    # Rank Messaging Theme by Purchases
    messaging_theme_ranking = rank_by_purchases(data, 'Messaging Theme')
    creative_theme_ranking = rank_by_purchases(data, 'Creative Theme')

    # Display rankings for Messaging Theme
    st.subheader("Messaging Theme Rankings (by Purchases)")
    for _, row in messaging_theme_ranking.iterrows():
        theme_value = row['Messaging Theme']
        st.write(f"**{theme_value}** - Purchases: {row['Purchases']}")
        
        # Get combination rankings with Creative Theme
        combo_rankings = rank_combinations(data, 'Messaging Theme', 'Creative Theme')
        filtered_combos = combo_rankings[combo_rankings['Messaging Theme'] == theme_value]

        # Display combinations DataFrame in the dropdown
        with st.expander(f"See combinations with Creative Theme for {theme_value}"):
            st.dataframe(filtered_combos)

    # Display rankings for Creative Theme
    st.subheader("Creative Theme Rankings (by Purchases)")
    for _, row in creative_theme_ranking.iterrows():
        theme_value = row['Creative Theme']
        st.write(f"**{theme_value}** - Purchases: {row['Purchases']}")
        
        # Get combination rankings with Messaging Theme
        combo_rankings = rank_combinations(data, 'Creative Theme', 'Messaging Theme')
        filtered_combos = combo_rankings[combo_rankings['Creative Theme'] == theme_value]

        # Display combinations DataFrame in the dropdown
        with st.expander(f"See combinations with Messaging Theme for {theme_value}"):
            st.dataframe(filtered_combos)
