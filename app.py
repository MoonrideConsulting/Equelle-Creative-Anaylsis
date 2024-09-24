import streamlit as st
import cross_section
import ranked_combos
import combo_breakdown
import machine_learning
from google.oauth2 import service_account
from google.cloud import bigquery

# Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="Equelle Creative Analysis", page_icon="🧑‍🚀", layout="wide")

# Function to load data from BigQuery
def load_data():
    if 'full_data' not in st.session_state:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(credentials=credentials)
        query = """
        SELECT *
        FROM `Equelle_Segments.equelle_ad_level_all`
        WHERE DATE(Date) > "2024-01-01";
        """
        st.session_state.full_data = pd.read_gbq(query, credentials=credentials)

# Main function to control navigation
def main_dashboard():
    # Load the data into session state
    load_data()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Ranked Combinations", "Cross Section Analysis", "Combo Breakdown", "Machine Learning Analysis"], index=0)

    # Set page header and layout
    st.markdown("<h1 style='text-align: center;'>Equelle Creative Analysis Dashboard</h1>", unsafe_allow_html=True)

    if page == "Ranked Combinations":
        ranked_combos.main()

    elif page == "Cross Section Analysis":
        cross_section.password_protection()

    elif page == "Combo Breakdown":
        combo_breakdown.main()

    elif page == "Machine Learning Analysis":
        machine_learning.main()

# Run the app
if __name__ == "__main__":
    main_dashboard()
