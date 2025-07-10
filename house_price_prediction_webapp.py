import numpy as np
import pickle
import streamlit as st

# Load the model
with open('house_price_prediction_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)

# Prediction function
def house_price_prediction(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    return f"Predicted House Price: ₹{prediction[0]:,.2f}"

def main():
    st.title('House Price Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        number_of_bedrooms = st.text_input('Number of Bedrooms')
    with col2:
        number_of_bathrooms = st.text_input('Number of Bathrooms')
    with col3:
        living_area = st.text_input('Living Area')
    with col1:
        lot_area = st.text_input('Lot Area')
    with col2:
        number_of_floors = st.text_input('Number of Floors')
    with col3:
        waterfront_present = st.text_input('Waterfront Present')
    with col1:
        number_of_views = st.text_input('Number of Views')
    with col2:
        condition_of_the_house = st.text_input('Condition of the House')
    with col3:
        grade_of_the_house = st.text_input('Grade of the House')
    with col1:
        area_of_house_excl_basement = st.text_input('Area of the House (excluding basement)')
    with col2:
        area_of_basement = st.text_input('Area of the Basement')
    with col3:
        built_year = st.text_input('Built Year')
    with col1:
        renovation_year = st.text_input('Renovation Year')
    with col2:
        postal_code = st.text_input('Postal Code')
    with col3:
        latitude = st.text_input('Latitude')
    with col1:
        longitude = st.text_input('Longitude')
    with col2:
        living_area_renov = st.text_input('Living Area after Renovation')
    with col3:
        lot_area_renov = st.text_input('Lot Area after Renovation')
    with col1:
        number_of_schools_nearby = st.text_input('Number of Schools Nearby')
    with col2:
        distance_from_airport = st.text_input('Distance from the Airport')

    house_price = ''

    if st.button('Predict House Price'):
        try:
            # Convert all inputs to float
            input_features = list(map(float, [
                number_of_bedrooms,
                number_of_bathrooms,
                living_area,
                lot_area,
                number_of_floors,
                waterfront_present,
                number_of_views,
                condition_of_the_house,
                grade_of_the_house,
                area_of_house_excl_basement,
                area_of_basement,
                built_year,
                renovation_year,
                postal_code,
                latitude,
                longitude,
                living_area_renov,
                lot_area_renov,
                number_of_schools_nearby,
                distance_from_airport
            ]))

            house_price = house_price_prediction(input_features)

        except ValueError:
            st.error("❌ Please enter valid numeric values for all fields.")

    if house_price:
        st.success(house_price)

if __name__ == '__main__':
    main()
