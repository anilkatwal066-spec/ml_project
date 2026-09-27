import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Titanic Survival Classification Dashboard",
    layout="wide"
)

df = pd.read_csv("classification/datasets/Titanic-Dataset.csv")
model = joblib.load("classification/titanic_model.pkl")

model_results = {
    "KNN": 0.81,
    "Logistic Regression": 0.81,
    "SVC": 0.81
}

best_model = max(model_results, key=model_results.get)
best_accuracy = model_results[best_model]

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Project Details",
        "Dataset",
        "EDA",
        "Model Comparison",
        "Prediction"
    ]
)

if page == "Project Details":

    st.title("Titanic Survival Classification Dashboard")

    st.write(
        "An interactive machine learning app to explore "
        "Titanic data and predict passenger survival."
    )

    st.header("Project Information")

    st.write("This is a machine learning classification project.")

    st.write(
        "The goal is to predict whether a passenger survived "
        "the Titanic disaster or not."
    )

    st.write("Dataset used: Titanic Dataset")
    st.write("Target column: Survived")

    st.write("Models used:")
    st.write("- KNN")
    st.write("- Logistic Regression")
    st.write("- SVC")

    st.header("Target Meaning")

    st.write("0 = Did Not Survive")
    st.write("1 = Survived")


elif page == "Dataset":

    st.title("Titanic Dataset")

    st.header("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Rows", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Statistical Summary")
    st.write(df.describe())


elif page == "EDA":

    st.title("Exploratory Data Analysis")

    chart = st.selectbox(
        "Select Visualization",
        [
            "Survival Count",
            "Survival by Gender",
            "Survival by Class",
            "Age Distribution"
        ]
    )

    if chart == "Survival Count":

        st.subheader("Survival Count")

        fig, ax = plt.subplots()

        sns.countplot(
            x="Survived",
            data=df,
            ax=ax
        )

        ax.set_xlabel("Survived")
        ax.set_ylabel("Number of Passengers")

        st.pyplot(fig)

    elif chart == "Survival by Gender":

        st.subheader("Survival by Gender")

        fig, ax = plt.subplots()

        sns.countplot(
            x="Sex",
            hue="Survived",
            data=df,
            ax=ax
        )

        ax.set_xlabel("Gender")
        ax.set_ylabel("Number of Passengers")

        st.pyplot(fig)

    elif chart == "Survival by Class":

        st.subheader("Survival by Passenger Class")

        fig, ax = plt.subplots()

        sns.countplot(
            x="Pclass",
            hue="Survived",
            data=df,
            ax=ax
        )

        ax.set_xlabel("Passenger Class")
        ax.set_ylabel("Number of Passengers")

        st.pyplot(fig)

    elif chart == "Age Distribution":

        st.subheader("Age Distribution")

        fig, ax = plt.subplots()

        sns.histplot(
            df["Age"].dropna(),
            bins=30,
            ax=ax
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Passengers")

        st.pyplot(fig)


elif page == "Model Comparison":

    st.title("Model Accuracy Comparison")

    results_df = pd.DataFrame(
        {
            "Model": list(model_results.keys()),
            "Accuracy": list(model_results.values())
        }
    )

    st.bar_chart(
        results_df.set_index("Model")
    )

    st.subheader("Model Accuracy")

    for name, accuracy in model_results.items():
        st.write(f"{name}: {accuracy:.2f}")

    st.success(
        f"Best Model: {best_model} with accuracy {best_accuracy:.2f}"
    )


elif page == "Prediction":

    st.title("Titanic Survival Prediction")

    st.write(
        "Enter the passenger information below "
        "to predict survival."
    )

    col1, col2 = st.columns(2)

    with col1:

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3]
        )

        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=25
        )

        sibsp = st.number_input(
            "Siblings/Spouses",
            min_value=0,
            max_value=8,
            value=0
        )

    with col2:

        parch = st.number_input(
            "Parents/Children",
            min_value=0,
            max_value=6,
            value=0
        )

        fare = st.number_input(
            "Fare",
            min_value=0.0,
            value=32.0
        )

        embarked = st.selectbox(
            "Embarked",
            ["C", "Q", "S"]
        )

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