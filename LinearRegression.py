import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.header('Salary Prediction by Perfection')

datafile = pd.read_csv('Salary Data.csv')


df = datafile[['Age', 'Gender', 'Educational_attainment', 'Experience Years', 'Salary']].dropna()

encoder_Gender = LabelEncoder()
df['Gender'] = encoder_Gender.fit_transform(df['Gender'])
encoder_Educational_attainment = LabelEncoder()
df['Educational_attainment'] = encoder_Educational_attainment.fit_transform(df['Educational_attainment'])


y = df[['Age', 'Gender', 'Educational_attainment', 'Experience Years']]
c = df[['Salary']] 

feature_train, feature_test, target_train, target_test = train_test_split(y, c, test_size=0.2)

model = LinearRegression()
model.fit(feature_train, target_train)

st.sidebar.header('Expected Salary For Jobs')
st.sidebar.subheader('Personal Details')
Age = st.sidebar.slider('select your age', 18, 64)
st.sidebar.write('I am:', Age)
Gender = st.sidebar.selectbox('Gender', encoder_Gender.classes_)
st.sidebar.write('I am a:', Gender)
Edu = st.sidebar.selectbox('Educational_attainment', encoder_Educational_attainment.classes_)
Exp = st.sidebar.number_input('Experience Years', min_value= 0)

# Encode user inputs using the trained encoders
Gender_encoded = encoder_Gender.transform([Gender])[0]
Education_encoded = encoder_Educational_attainment.transform([Edu])[0]


features = {'Age': [Age],
            'Gender': [Gender_encoded],
            'Educational_attainment': [Education_encoded],
            'Experience Years': [Exp]}

input_features = pd.DataFrame(features)
st.write('Personal Details:', input_features)


if st.button('Prediction on Salary'):
    prediction = model.predict(input_features)
    st.write(f'Your predicted salary based on the information you have given us is ${prediction[0][0]:,.2f}')
