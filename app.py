import streamlit as st

st.set_page_config(page_title="Equelle Creative Analysis", page_icon="🧑‍🚀", layout="wide")

import cross_section
import ranked_combos
import combo_breakdown
import machine_learning



# Main function to control navigation
def main_dashboard():
 
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
