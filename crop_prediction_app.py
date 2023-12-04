
# importing all the required libraries
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import gdown

# Accessing and Downloading the model from the google drive
model_url = f'https://drive.google.com/uc?id=1ANCwS3prrYEVXSloDSxvnRL1qcuIj3Xb'
output = 'crop_production_model.joblib'
gdown.download(model_url, output, quiet=False)

# Loading the model from the downloaded file
model = joblib.load(output)

# Loading the label encoders
label_encoders = {
    'le_State_Name': joblib.load("le_State_Name.joblib"),
    'le_District_Name': joblib.load("le_District_Name.joblib"),
    'le_Season': joblib.load("le_Season.joblib"),
    'le_Crop': joblib.load("le_Crop.joblib")
}

# Function to predict crop production
def predict_production(state, district, year, season, crop, area):
    # Encoding the input data using the loaded label encoders
    state_encoded = label_encoders['le_State_Name'].transform([state])[0]
    district_encoded = label_encoders['le_District_Name'].transform([district])[0]
    season_encoded = label_encoders['le_Season'].transform([season])[0]
    crop_encoded = label_encoders['le_Crop'].transform([crop])[0]

    # Making predictions using the loaded model
    input_data = np.array([[state_encoded, district_encoded, year, season_encoded, crop_encoded, area]])
    input_data = input_data.astype(float)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app layout
st.title("Crop Production Prediction App")

# Loading the dataset 
df = pd.read_csv("crop_production.csv")

# Taking input from the user
state = st.selectbox("Select State", df['State_Name'].unique())

# Filter districts based on the selected state
filtered_districts = df[df['State_Name'] == state]['District_Name'].unique()
district = st.selectbox("Select District", filtered_districts)

# Filter seasons based on the selected state and district
filtered_seasons = df[(df['State_Name'] == state) & (df['District_Name'] == district)]['Season'].unique()
season = st.selectbox("Select Season", filtered_seasons)

# Filter crops based on the selected state, district, and season
filtered_crops = df[(df['State_Name'] == state) & (df['District_Name'] == district) & (df['Season'] == season)]['Crop'].unique()
crop = st.selectbox("Select Crop", filtered_crops)

# Allowing the user to input any year and area
year = st.number_input("Enter Year", min_value=int(df['Crop_Year'].min()), value=int(df['Crop_Year'].median()))
area = st.number_input("Enter Area", min_value=float(df['Area'].min()), value=float(df['Area'].median()))

# Button to trigger prediction
if st.button("Predict Production"):
    prediction = predict_production(state, district, year, season, crop, area)
    st.success(f'Predicted Production: {prediction:.2f}')
