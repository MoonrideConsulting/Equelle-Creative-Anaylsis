import streamlit as st
import cross_section
import ranked_combos
import combo_breakdown
import machine_learning

st.set_page_config(page_title="Equelle Creative Analysis", page_icon="🧑‍🚀", layout="wide")


# Main function to control navigation
def main_dashboard():

    if 'full_data' not in st.session_state:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = bigquery.Client(credentials=credentials)
        query = f"""
        SELECT *
        FROM `Equelle_Segments.equelle_ad_level_all`
        WHERE DATE(Date) >= "2024-01-01";"""
        st.session_state.full_data = pandas.read_gbq(query, credentials=credentials)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Cross Section Analysis", "Ranked Combinations", "Combo Breakdown", "Machine Learning Analysis"], index=0)

    # Set page header and layout
    st.markdown("<h1 style='text-align: center;'>Equelle Creative Analysis Dashboard</h1>", unsafe_allow_html=True)

    if page == "Cross Section Analysis":
        st.markdown("<h2 style='text-align: center;'>Cross Section Analysis</h2>", unsafe_allow_html=True)
        cross_section.password_protection()  # Call the main() function from cross_section_analysis.py
    
    elif page == "Ranked Combinations":
        st.markdown("<h2 style='text-align: center;'>Ranked Combinations</h2>", unsafe_allow_html=True)
        ranked_combos.main()

    elif page == "Combo Breakdown":
        st.markdown("<h2 style='text-align: center;'>Combo Breakdown</h2>", unsafe_allow_html=True)
        combo_breakdown.main()

    elif page == "Machine Learning Analysis":
        st.markdown("<h2 style='text-align: center;'>Machine Learning Analysis</h2>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Find variables/combos that the model deems more relevant in changing an ad's purchase volume.</h3>", unsafe_allow_html=True)
        machine_learning.main()

# Run the app with password protection
if __name__ == "__main__":
    main_dashboard()  # Call the main dashboard if the password is correct
