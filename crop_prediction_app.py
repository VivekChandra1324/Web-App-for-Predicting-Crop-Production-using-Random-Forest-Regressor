import streamlit as st
import numpy as np
import joblib
import pandas as pd  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# Loading  the trained model and label encoders
model = joblib.load("crop_production_model.joblib")
label_encoders = {
    'le_State_Name': joblib.load("le_State_Name.joblib"),
    'le_District_Name': joblib.load("le_District_Name.joblib"),
    'le_Season': joblib.load("le_Season.joblib"),
    'le_Crop': joblib.load("le_Crop.joblib")
}


df = pd.read_csv("C:/Users/user/Informatics/Project/crop_production.csv")

state_counts= df['State_Name'].value_counts()
# removing states with value counts less than 1000
df = df[df['State_Name'].isin(state_counts[state_counts >= 1000].index)]
year_counts= df['Crop_Year'].value_counts()
# filtering year with value counts less than 100
df = df[df['Crop_Year'].isin(year_counts[year_counts >= 100].index)]
area_counts=df['Area'].value_counts()
# filtering area with value counts less than 12
df = df[df['Area'].isin(area_counts[area_counts >= 12].index)]
crop_counts=df['Crop'].value_counts()
# filtering crop with value counts less than 1000
df = df[df['Crop'].isin(crop_counts[crop_counts >= 1000].index)]
exclude_crops = ["Other  Rabi pulses", "Other Kharif pulses"]
df = df[~df['Crop'].isin(exclude_crops)]
district_counts=df['District_Name'].value_counts()
# filtering district with value counts less than 200
df = df[df['District_Name'].isin(district_counts[district_counts >= 200].index)]
season_counts= df['Season'].value_counts()
# filtering season with value counts less than 1000
df = df[df['Season'].isin(season_counts[season_counts >= 1000].index)]


def predict_production(state, district, year, season, crop, area):
    try:
        # Applying label encoding to user input
        state_encoded = label_encoders['le_State_Name'].transform([state])[0]
        district_encoded = label_encoders['le_District_Name'].transform([district])[0]
        season_encoded = label_encoders['le_Season'].transform([season])[0]
        crop_encoded = label_encoders['le_Crop'].transform([crop])[0]

        # Make predictions using the trained model
        input_data = np.array([[state_encoded, district_encoded, year, season_encoded, crop_encoded, area]])
        prediction = model.predict(input_data)[0]

        return prediction

    except ValueError as ve:
        if "y contains previously unseen labels" in str(ve):
            st.error("Error: New label encountered. Please make sure the selected value is present in the training dataset.")
        else:
            st.error(f"An error occurred: {str(ve)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None


# Streamlit app layout
st.title("Crop Production Prediction App")

# Taking input form User
state = st.selectbox("Select State", df['State_Name'].unique())
district = st.selectbox("Select District", df['District_Name'].unique())
year = st.number_input("Enter Year", min_value=int(df['Crop_Year'].min()), max_value=int(df['Crop_Year'].max()), value=int(df['Crop_Year'].median()))
season = st.selectbox("Select Season", df['Season'].unique())
crop = st.selectbox("Select Crop", df['Crop'].unique())
area = st.number_input("Enter Area", min_value=float(df['Area'].min()), max_value=float(df['Area'].max()), value=float(df['Area'].median()))

# Button to trigger prediction
if st.button("Predict Production"):
    prediction = predict_production(state, district, year, season, crop, area)
    if prediction is not None:
        st.success(f'Predicted Production: {prediction:.2f}')
