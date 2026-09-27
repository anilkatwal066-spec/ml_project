import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Insurance Charges Prediction",
    layout="wide"
)

df = pd.read_csv(
    r"F:\MLPROJECT\New folder\dataset\insurance.csv"
)

model = joblib.load(
    r"F:\MLPROJECT\New folder\insurance_model.pkl"
)

st.sidebar.title("Insurance Prediction")
page = st.sidebar.radio(
    "Select a page",
    [
        "Home",
        "Dataset",
        "Visualizations",
        "Model Performance",
        "Prediction"
    ]
)

if page == "Home":

    st.title("Insurance Charges Prediction")

    st.write(
        "This project uses machine learning to predict "
        "medical insurance charges."
    )

    st.write(
        "The prediction is based on information such as age, "
        "BMI, number of children, smoking status, sex and region."
    )

    st.subheader("Project Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Rows", df.shape[0])
    col2.metric("Dataset Columns", df.shape[1])
    col3.metric("Model", "Linear Regression")

    st.subheader("What this project does")

    st.write(
        "1. Loads and explores the insurance dataset."
    )

    st.write(
        "2. Looks at relationships between different variables."
    )

    st.write(
        "3. Trains a Linear Regression model."
    )

    st.write(
        "4. Uses the trained model to predict insurance charges."
    )


elif page == "Dataset":

    st.title("Insurance Dataset")

    st.write(
        "Here you can explore the data used in this project."
    )

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.subheader("Dataset Preview")

    st.dataframe(df)

    st.subheader("Filter Data")

    selected_smoker = st.selectbox(
        "Smoking status",
        ["All"] + list(df["smoker"].unique())
    )

    if selected_smoker == "All":
        filtered_df = df
    else:
        filtered_df = df[df["smoker"] == selected_smoker]

    st.write(
        "Showing",
        len(filtered_df),
        "rows"
    )

    st.dataframe(filtered_df)

    st.subheader("Basic Statistics")

    st.write(df.describe())


elif page == "Visualizations":

    st.title("Visualizations")

    chart = st.selectbox(
        "Choose a visualization",
        [
            "Insurance Charges Distribution",
            "Age vs Charges",
            "BMI vs Charges",
            "Smoker vs Charges",
            "Region vs Charges"
        ]
    )

    if chart == "Insurance Charges Distribution":

        st.subheader("Distribution of Insurance Charges")

        fig, ax = plt.subplots()

        sns.histplot(
            df["charges"],
            bins=30,
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Charges")
        ax.set_ylabel("Number of People")

        st.pyplot(fig)

    elif chart == "Age vs Charges":

        st.subheader("Age and Insurance Charges")

        fig, ax = plt.subplots()

        sns.scatterplot(
            data=df,
            x="age",
            y="charges",
            ax=ax
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Charges")

        st.pyplot(fig)

    elif chart == "BMI vs Charges":

        st.subheader("BMI and Insurance Charges")

        fig, ax = plt.subplots()

        sns.scatterplot(
            data=df,
            x="bmi",
            y="charges",
            ax=ax
        )

        ax.set_xlabel("BMI")
        ax.set_ylabel("Charges")

        st.pyplot(fig)

    elif chart == "Smoker vs Charges":

        st.subheader("Smoking Status and Charges")

        fig, ax = plt.subplots()

        sns.boxplot(
            data=df,
            x="smoker",
            y="charges",
            ax=ax
        )

        ax.set_xlabel("Smoker")
        ax.set_ylabel("Charges")

        st.pyplot(fig)

    elif chart == "Region vs Charges":

        st.subheader("Region and Insurance Charges")

        fig, ax = plt.subplots()

        sns.boxplot(
            data=df,
            x="region",
            y="charges",
            ax=ax
        )

        ax.set_xlabel("Region")
        ax.set_ylabel("Charges")

        st.pyplot(fig)


elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "The model used for this project is Linear Regression."
    )

    X = df.drop("charges", axis=1)
    y = df["charges"]

    X = pd.get_dummies(X, dtype=int)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")

    st.subheader("What do these numbers mean?")

    st.write(
        "MAE shows the average difference between the actual "
        "and predicted charges."
    )

    st.write(
        "RMSE also measures prediction error, but gives more "
        "importance to larger errors."
    )

    st.write(
        "R² shows how well the model explains the changes "
        "in insurance charges."
    )


elif page == "Prediction":

    st.title("Predict Insurance Charges")

    st.write(
        "Enter the information below and the model will "
        "estimate the insurance charges."
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=30
        )

        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0
        )

    with col2:

        children = st.number_input(
            "Number of children",
            min_value=0,
            max_value=10,
            value=0
        )

        smoker = st.selectbox(
            "Smoker",
            ["yes", "no"]
        )

        region = st.selectbox(
            "Region",
            [
                "southwest",
                "southeast",
                "northwest",
                "northeast"
            ]
        )

    if st.button("Predict"):

        input_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        })

        input_data = pd.get_dummies(
            input_data,
            dtype=int
        )

        training_data = df.drop(
            "charges",
            axis=1
        )

        training_data = pd.get_dummies(
            training_data,
            dtype=int
        )

        input_data = input_data.reindex(
            columns=training_data.columns,
            fill_value=0
        )

        prediction = model.predict(input_data)

        st.success(
            f"Estimated insurance charges: ${prediction[0]:,.2f}"
        )