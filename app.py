import streamlit as st
import cross_section
import ranked_combos
import machine_learning

# Main function to control navigation
def main_dashboard():

    st.set_page_config(page_title="Equelle Creative Analysis",page_icon="🧑‍🚀",layout="wide")

    st.markdown("<h1 style='text-align: center;'>Equelle Creative Analysis</h1>", unsafe_allow_html=True)
    # Set up navigation for different pages (using radio buttons for tabs)
    page = st.radio("Select a page", ["Cross Section Analysis", "Ranked Combinations", "Machine Learning Analyis"], index = 0)

    if page == "Cross Section Analysis":
        st.markdown("<h2 style='text-align: center;'>Cross Section Analysis</h2>", unsafe_allow_html=True)
        cross_section.password_protection()  # Call the main() function from cross_section_analysis.py
    
    elif page == "Ranked Combinations":
        st.markdown("<h2 style='text-align: center;'>Ranked Combinations</h2>", unsafe_allow_html=True)
        ranked_combos.main()

    elif page == "Machine Learning Analyis":
        st.markdown("<h2 style='text-align: center;'>Machine Learning Analyis</h2>", unsafe_allow_html=True)
        machine_learning.main()

# Run the dashboard
if __name__ == "__main__":
    main_dashboard()
