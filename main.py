# This is a web application that diagnosis breast cancer with machine learning and Python

# Import the libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import requests
from io import BytesIO
import s3fs
import streamlit as st

# Create a title and sub-title
st.write("""
# Breast Cancer Detection
Classify benign and malignant tumors to determine if someone has breast cancer using ML and Python!
""")

# Open and display thumbnail image
response = requests.get('https://breast-cancer-assets.s3-us-west-1.amazonaws.com/breastCancerThumbnail.png')
image = Image.open(BytesIO(response.content))
st.image(image, caption='ML', use_column_width=True)

# Get the data
df = pd.read_csv('s3://breast-cancer-assets/breastcancerdata.csv')

# Set a sub header
st.subheader('Data Information:')

# Show the data as a table
st.dataframe(df)

# Show statistics on the data
st.write(df.describe())

# Show the data as a chart
chart = st.bar_chart(df)

# Create the feature and target data set
X = df.drop(['diagnosis'], 1)
Y = np.array(df['diagnosis'])

# Split the data set into 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Transform the feature data to be values between 0 and 1
sc = MinMaxScaler(feature_range=(0,1))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Get the feature input from the user
def get_user_input():
    radius_mean = st.sidebar.slider('radius_mean', 3.0, 35.0, 10.0)
    texture_mean = st.sidebar.slider('texture_mean', 5.0, 45.0, 25.0)
    perimeter_mean = st.sidebar.slider('perimeter_mean', 35.0, 205.0, 115.0)
    area_mean = st.sidebar.slider('area_mean', 125.0, 2555.0, 1000.0)
    smoothness_mean = st.sidebar.slider('smoothness_mean', 0.0, 5.0, 2.5)
    compactness_mean = st.sidebar.slider('compactness_mean', 0.0, 5.0, 2.5)
    concavity_mean = st.sidebar.slider('concavity_mean', 0.0, 5.0, 2.5)
    concave_points_mean = st.sidebar.slider('concave_points_mean', 0.0, 5.0, 2.5)
    symmetry_mean = st.sidebar.slider('symmetry_mean', 0.0, 5.0, 2.5)
    fractal_dimension_mean = st.sidebar.slider('fractal_dimension_mean', 0.0, 5.0, 2.5)
    radius_se = st.sidebar.slider('radius_se', 0.0, 10.0, 5.0)
    texture_se = st.sidebar.slider('texture_se', 0.0, 10.0, 5.0)
    perimeter_se = st.sidebar.slider('perimeter_se', 0.0, 30.0, 15.0)
    area_se = st.sidebar.slider('area_se', 0.0, 550.0, 225.0)
    smoothness_se = st.sidebar.slider('smoothness_se', 0.0, 5.0, 2.5)
    compactness_se = st.sidebar.slider('compactness_se', 0.0, 5.0, 2.5)
    concavity_se = st.sidebar.slider('concavity_se', 0.0, 5.0, 2.5)
    concave_points_se = st.sidebar.slider('concave_points_se', 0.0, 5.0, 2.5)
    symmetry_se = st.sidebar.slider('symmetry_se', 0.0, 5.0, 2.5)
    fractal_dimension_se = st.sidebar.slider('fractal_dimension_se', 0.0, 5.0, 2.5)
    radius_worst = st.sidebar.slider('radius_worst', 2.5, 45.0, 25.0)
    texture_worst = st.sidebar.slider('texture_worst', 5.0, 65.0, 30.0)
    perimeter_worst = st.sidebar.slider('perimeter_worst', 35.0, 275.0, 125.0)
    area_worst = st.sidebar.slider('area_worst', 150.0, 4350.0, 1925.0)
    smoothness_worst = st.sidebar.slider('smoothness_worst', 0.0, 5.0, 2.5)
    compactness_worst = st.sidebar.slider('compactness_worst', 0.0, 5.0, 2.5)
    concavity_worst = st.sidebar.slider('concavity_worst', 0.0, 5.0, 2.5)
    concave_points_worst = st.sidebar.slider('concave_points_worst', 0.0, 5.0, 2.5)
    symmetry_worst = st.sidebar.slider('symmetry_worst', 0.0, 5.0, 2.5)
    fractal_dimension_worst = st.sidebar.slider('fractal_dimension_worst', 0.0, 5.0, 2.5)

    # Store a dictionary into a variable
    user_data = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'concave_points_mean': concave_points_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'perimeter_se': perimeter_se,
        'area_se': area_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave_points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'radius_worst': radius_worst,
        'texture_worst': texture_worst,
        'perimeter_worst': perimeter_worst,
        'area_worst': area_worst,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'concave_points_worst': concave_points_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst,
    }

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the users' input
st.subheader('User Input:')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the models' metrics
st.subheader('Model Test Accuracy Score:')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%' )

# Store the models' predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)

diagnosis_certainty = str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%'

if prediction == 1:
    diagnosis_statement = "There is a {} chance you have breast cancer. God bless you <3".format(diagnosis_certainty)
elif prediction == 0:
    diagnosis_statement = "There is a {} chance you do not have breast cancer. God bless you <3".format(diagnosis_certainty)

st.write(diagnosis_statement)