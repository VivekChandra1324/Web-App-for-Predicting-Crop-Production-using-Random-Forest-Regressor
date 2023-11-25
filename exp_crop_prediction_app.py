import streamlit as st
import numpy as np
import joblib
import pandas as pd
import gdown

# Loading the trained model and label encoders
#model = joblib.load("https://drive.google.com/file/d/1wBX1fJbpQMpBGPfLLaLSHIcUGGq5RTQv/view?usp=drive_link/crop_production_model.joblib")


# Provide the Google Drive file ID for your model file
#file_id = '1wBX1fJbpQMpBGPfLLaLSHIcUGGq5RTQv'
#model_url = f'https://drive.google.com/uc?id={file_id}'
model_url = f'https://drive.google.com/uc?id=1wBX1fJbpQMpBGPfLLaLSHIcUGGq5RTQv'

# Download the model file
output = 'crop_production_model.joblib'
gdown.download(model_url, output, quiet=False)

# Load the model from the downloaded file
model = joblib.load(output)











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

# Loading the dataset for dropdown options
df = pd.read_csv("crop_production.csv")

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





# Taking input from the user
state = st.selectbox("Select State", df['State_Name'].unique())
filtered_districts = df[df['State_Name'] == state]['District_Name'].unique()
district = st.selectbox("Select District", filtered_districts)
filtered_crops = df[df['District_Name'] == district]['Crop'].unique()
crop = st.selectbox("Select Crop", filtered_crops)

# Filter seasons based on the selected district and crop
filtered_seasons = df[(df['District_Name'] == district) & (df['Crop'] == crop)]['Season'].unique()
season = st.selectbox("Select Season", filtered_seasons)

# Allow the user to input any year and area
year = st.number_input("Enter Year", min_value=int(df['Crop_Year'].min()), value=int(df['Crop_Year'].median()))
area = st.number_input("Enter Area", min_value=float(df['Area'].min()), value=float(df['Area'].median()))

# Button to trigger prediction
if st.button("Predict Production"):
    prediction = predict_production(state, district, year, season, crop, area)
    st.success(f'Predicted Production: {prediction:.2f}')
