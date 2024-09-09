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
        grouped['CPM'] = (grouped['Spend'] / grouped['Impressions']) * 1000
        grouped['CPA'] = grouped['Spend'] / grouped['Purchases']
        grouped['CPC'] = grouped['Spend'] / grouped['Clicks']

        # Combine the values in the columns to create a 'Combination' identifier
        grouped['Combination'] = grouped.apply(lambda row: ', '.join([f"{col}={row[col]}" for col in combo]), axis=1)

        # Append the results to the combined dataframe
        combined_results = pd.concat([combined_results, grouped[['Combination', 'Spend', 'Clicks', 'Impressions', 'Purchases', 'CPM', 'CPA', 'CPC']]])

    # Sort the results by Purchases in descending order
    combined_results = combined_results.sort_values(by='Purchases', ascending=False)

    return combined_results


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
    st.write(data.columns)

    num_combos = st.number_input("Pick a number", 2, 4)

    st.dataframe(cross_section_analysis(data, num_combos), use_container_width=True)


password_protection()
