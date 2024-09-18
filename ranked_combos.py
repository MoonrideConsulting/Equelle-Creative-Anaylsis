import streamlit as st
import pandas as pd

# Function to rank each value of a variable based on Purchases
def rank_by_purchases(data, column):
    ranking = data.groupby(column).agg({
        'Purchases': 'sum'
    }).reset_index().sort_values(by='Purchases', ascending=False)

    return ranking

# Function to get combination rankings for each value of the variable
def rank_combinations(data, main_column, secondary_column):
    # Group by the combination of main_column and secondary_column
    combo_rankings = data.groupby([main_column, secondary_column]).agg({
        'Purchases': 'sum'
    }).reset_index().sort_values(by='Purchases', ascending=False)

    return combo_rankings

# Main function to display ranked combos
def main():
    st.markdown("<h1 style='text-align: center;'>Ranked Combos</h1>", unsafe_allow_html=True)

    # Load the data from session state (assuming it's already loaded)
    data = st.session_state.full_data
    
    # Prepare the data (assuming relevant columns like 'Messaging Theme' and 'Creative Theme' exist)
    # You can adjust this part to your specific dataset
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
        
        # Dropdown for combination rankings with Creative Theme
        st.write(f"Combinations with Creative Theme for {theme_value}:")
        combo_rankings = rank_combinations(data, 'Messaging Theme', 'Creative Theme')
        filtered_combos = combo_rankings[combo_rankings['Messaging Theme'] == theme_value]

        # Display combinations in a dropdown
        selected_combo = st.selectbox(f"Select combination for {theme_value}:", filtered_combos['Creative Theme'], key=f"combo_{theme_value}")
        combo_info = filtered_combos[filtered_combos['Creative Theme'] == selected_combo]
        st.write(f"Selected Combination: {theme_value} and {selected_combo} - Purchases: {combo_info['Purchases'].values[0]}")

    # Display rankings for Creative Theme
    st.subheader("Creative Theme Rankings (by Purchases)")
    for _, row in creative_theme_ranking.iterrows():
        theme_value = row['Creative Theme']
        st.write(f"**{theme_value}** - Purchases: {row['Purchases']}")
        
        # Dropdown for combination rankings with Messaging Theme
        st.write(f"Combinations with Messaging Theme for {theme_value}:")
        combo_rankings = rank_combinations(data, 'Creative Theme', 'Messaging Theme')
        filtered_combos = combo_rankings[combo_rankings['Creative Theme'] == theme_value]

        # Display combinations in a dropdown
        selected_combo = st.selectbox(f"Select combination for {theme_value}:", filtered_combos['Messaging Theme'], key=f"combo_{theme_value}_2")
        combo_info = filtered_combos[filtered_combos['Messaging Theme'] == selected_combo]
        st.write(f"Selected Combination: {theme_value} and {selected_combo} - Purchases: {combo_info['Purchases'].values[0]}")
Key Features:
Rankings for Messaging and Creative Themes:

The function rank_by_purchases() ranks the values of Messaging Theme and Creative Theme based on total purchases.
Dropdowns for Combination Rankings:

For each value of the ranked theme (e.g., Messaging Theme or Creative Theme), we add a dropdown that lets the user select combinations with the other theme (e.g., combinations of Messaging Theme with Creative Theme).
This is done using rank_combinations(), which ranks combinations of the two variables by purchases.
Dropdowns with Ranking Information:

When a user selects a combination (e.g., a Messaging Theme with a Creative Theme), the app shows the number of purchases for that combination.
Integration with Main App:
In your main.py, you can add a new page that calls this Ranked Combos page:

python
Copy code
import streamlit as st
import Cross_section
import ranked_combos

# Main function to control navigation
def main_dashboard():
    # Set up navigation for different pages (using radio buttons for tabs)
    page = st.radio("Select a page", ["Cross Section Analysis", "Ranked Combos"], index=0)

    if page == "Cross Section Analysis":
        st.markdown("<h1 style='text-align: center;'>Cross Section Analysis</h1>", unsafe_allow_html=True)
        Cross_section.password_protection()

    elif page == "Ranked Combos":
        ranked_combos.main()  # Call the Ranked Combos page
