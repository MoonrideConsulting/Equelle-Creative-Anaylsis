import streamlit as st
import Cross_section

# Main function to control navigation
def main_dashboard():
    # Set up navigation for different pages (using radio buttons for tabs)
    page = st.radio("Select a page", ["Overview", "Cross Section Analysis", "ML Analysis"])

    if page == "Overview":
        st.markdown("<h1 style='text-align: center;'>Overview</h1>", unsafe_allow_html=True)
        st.write("This is the overview page where you can provide general insights, summaries, or other key metrics.")
        st.write("Add charts or summaries here...")

    elif page == "Cross Section Analysis":
        st.markdown("<h1 style='text-align: center;'>Cross Section Analysis</h1>", unsafe_allow_html=True)
        cross_section_analysis.password_protection()  # Call the main() function from cross_section_analysis.py

# Run the dashboard
if __name__ == "__main__":
    main_dashboard()
