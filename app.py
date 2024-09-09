import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pandas_gbq
import pandas 
from google.oauth2 import service_account
from google.cloud import bigquery
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="Equelle Creative Analysis",page_icon="🧑‍🚀",layout="wide")

def password_protection():
        main_dashboard()

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
    st.write(data)

password_protection()
