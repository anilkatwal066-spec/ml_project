import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("classification/titanic_model.pkl")


st.title("Titanic Survival Prediction")

st.subheader("Titanic Dataset")

uploaded_file = st.file_uploader("Upload Titanic CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write(df.head())

    chart = st.selectbox(
        "Select Visualization",
        ["Survival Count", "Survival by Gender", "Survival by Class", "Age Distribution"]
    )

    if chart == "Survival Count":
        fig, ax = plt.subplots()
        sns.countplot(x="Survived", data=df, ax=ax)
        st.pyplot(fig)

    elif chart == "Survival by Gender":
        fig, ax = plt.subplots()
        sns.countplot(x="Sex", hue="Survived", data=df, ax=ax)
        st.pyplot(fig)

    elif chart == "Survival by Class":
        fig, ax = plt.subplots()
        sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax)
        st.pyplot(fig)

    elif chart == "Age Distribution":
        fig, ax = plt.subplots()
        sns.histplot(df["Age"].dropna(), bins=30, ax=ax)
        st.pyplot(fig)


st.subheader("Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=8, value=0)
parch = st.number_input("Parents/Children", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

if st.button("Predict"):
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Sex_female": [1 if sex == "female" else 0],
        "Sex_male": [1 if sex == "male" else 0],
        "Embarked_C": [1 if embarked == "C" else 0],
        "Embarked_Q": [1 if embarked == "Q" else 0],
        "Embarked_S": [1 if embarked == "S" else 0]
    })

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger is predicted to survive.")
    else:
        st.error("Passenger is predicted not to survive.")
