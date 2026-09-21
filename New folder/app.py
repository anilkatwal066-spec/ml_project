import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Data Analysis App", layout="wide")

st.title("Data Analysis App")

file = st.file_uploader("Upload a CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("Data Preview")
    st.dataframe(df)

    st.sidebar.header("Controls")

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    # Dashboard
    st.subheader("Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    with col3:
        st.metric("Numeric Columns", len(numeric_columns))

    with col4:
        st.metric("Missing Values", int(df.isnull().sum().sum()))

    chart_type = st.sidebar.selectbox(
        "Select Chart Type",
        ["Line Chart", "Bar Chart", "Pie Chart"]
    )

    x_column = st.sidebar.selectbox(
        "Select X-axis Column",
        df.columns
    )

    y_columns = [col for col in numeric_columns if col != x_column]

    if y_columns:
        y_column = st.sidebar.selectbox(
            "Select Y-axis Column",
            y_columns
        )

        st.subheader("Visualization")

        data = df[[x_column, y_column]].dropna()

        if chart_type == "Line Chart":
            st.line_chart(data.set_index(x_column))

        elif chart_type == "Bar Chart":
            st.bar_chart(data.set_index(x_column))

        elif chart_type == "Pie Chart":
            fig, ax = plt.subplots()

            data.groupby(x_column)[y_column].sum().plot(
                kind="pie",
                autopct="%1.1f%%",
                ax=ax
            )

            ax.set_ylabel("")
            st.pyplot(fig)

    else:
        st.warning("Please select a different X-axis column.")


# Load trained model
model = joblib.load(r"F:\mlproject\New folder\insurance_model.pkl")


# Insurance prediction
st.subheader("Insurance Charges Prediction")

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
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

children = st.number_input(
    "Children",
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
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)


if st.button("Predict"):

    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],

        "sex_female": [
            1 if sex == "female" else 0
        ],

        "sex_male": [
            1 if sex == "male" else 0
        ],

        "smoker_no": [
            1 if smoker == "no" else 0
        ],

        "smoker_yes": [
            1 if smoker == "yes" else 0
        ],

        "region_northeast": [
            1 if region == "northeast" else 0
        ],

        "region_northwest": [
            1 if region == "northwest" else 0
        ],

        "region_southeast": [
            1 if region == "southeast" else 0
        ],

        "region_southwest": [
            1 if region == "southwest" else 0
        ]
    })

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Insurance Charges: ${prediction[0]:,.2f}"
    )