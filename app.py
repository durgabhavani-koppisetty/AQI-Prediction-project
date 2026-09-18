import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AQI Prediction",
    page_icon="🌍",
    layout="wide"
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "city_day.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_aqi_model.pkl"
)

PLOTS_DIR = os.path.join(
    BASE_DIR,
    "plots"
)


# =========================================================
# CHECK FILES
# =========================================================

if not os.path.exists(DATA_PATH):
    st.error(
        f"Dataset not found.\n\nExpected location:\n{DATA_PATH}"
    )
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model not found.\n\nExpected location:\n{MODEL_PATH}"
    )
    st.stop()


# =========================================================
# LOAD DATA AND MODEL
# =========================================================

try:
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

except Exception as e:
    st.error(f"Error loading dataset or model: {e}")
    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("🌍 Air Quality Index (AQI) Prediction")

st.subheader("Machine Learning Regression Project")

st.write(
    "This application predicts Air Quality Index (AQI) using "
    "air pollution parameters and a Random Forest Regression model."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Dataset",
        "Data Analysis",
        "Models & Evaluation",
        "Charts",
        "AQI Prediction"
    ]
)


# =========================================================
# 1. PROJECT OVERVIEW
# =========================================================

if page == "Project Overview":

    st.header("📌 Project Overview")

    st.write("""
    The AQI Prediction project uses Machine Learning regression
    techniques to predict the Air Quality Index from different
    air pollution parameters.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Rows",
            f"{df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Dataset Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Number of Cities",
            df["City"].nunique()
        )

    st.subheader("🎯 Objective")

    st.write("""
    The main objective is to develop a Machine Learning model
    capable of predicting AQI based on pollutant concentrations,
    city and date-related information.
    """)

    st.subheader("🔬 Machine Learning Type")

    st.write("**Regression**")

    st.subheader("📊 Input Parameters")

    st.write("""
    • City

    • PM2.5

    • PM10

    • NO

    • NO2

    • NOx

    • NH3

    • CO

    • SO2

    • O3

    • Benzene

    • Toluene

    • Xylene

    • Year

    • Month

    • Day

    • Day of Week
    """)

    st.subheader("🧹 Data Cleaning")

    st.write("""
    • Rows without AQI values were removed.

    • AQI_Bucket was removed because it is derived from AQI
      and could cause target leakage.

    • Date was converted into Year, Month, Day and DayOfWeek.

    • Missing numerical values were handled using median
      imputation.

    • City was encoded using One-Hot Encoding.
    """)


# =========================================================
# 2. DATASET
# =========================================================

elif page == "Dataset":

    st.header("📂 Dataset")

    st.write(
        "Dataset used: Air Quality Data in India"
    )

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    st.write("**Columns:**")

    st.write(
        list(df.columns)
    )

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.subheader("Cities")

    st.write(
        f"Number of cities: **{df['City'].nunique()}**"
    )

    st.write(
        sorted(df["City"].dropna().unique())
    )


# =========================================================
# 3. DATA ANALYSIS
# =========================================================

elif page == "Data Analysis":

    st.header("📈 Data Analysis")

    st.subheader("AQI Statistics")

    st.dataframe(
        df["AQI"].describe().to_frame("AQI"),
        use_container_width=True
    )

    st.subheader("AQI Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["AQI"].dropna(),
        bins=50
    )

    ax.set_title("AQI Distribution")
    ax.set_xlabel("AQI")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("AQI Boxplot")

    fig, ax = plt.subplots(figsize=(10, 3))

    ax.boxplot(
        df["AQI"].dropna(),
        vert=False
    )

    ax.set_xlabel("AQI")
    ax.set_title("AQI Boxplot")

    st.pyplot(fig)

    plt.close(fig)

    st.subheader("Correlation with AQI")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    correlation = (
        numeric_df
        .corr()["AQI"]
        .sort_values(ascending=False)
        .to_frame("Correlation with AQI")
    )

    st.dataframe(
        correlation,
        use_container_width=True
    )

    st.subheader("Top Correlated Pollutants")

    st.write("""
    The correlation analysis shows the relationship between
    pollutant variables and AQI in the dataset.
    """)


# =========================================================
# 4. MODELS & EVALUATION
# =========================================================

elif page == "Models & Evaluation":

    st.header("🤖 Machine Learning Models")

    st.write("""
    Four regression models were trained and evaluated using
    the same train-test split.
    """)

    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost"
        ],
        "MAE": [
            29.967730,
            20.340423,
            23.207873,
            20.659653
        ],
        "MSE": [
            3214.070355,
            1602.560313,
            1898.687480,
            1767.107321
        ],
        "RMSE": [
            56.692772,
            40.031991,
            43.573931,
            42.036976
        ],
        "R²": [
            0.824473,
            0.912481,
            0.896309,
            0.903495
        ]
    })

    st.subheader("Model Comparison")

    st.dataframe(
        results,
        use_container_width=True
    )

    st.subheader("Model Details")

    st.write("### 1. Linear Regression")

    st.write(
        "A basic regression model used as a baseline model."
    )

    st.write("### 2. Random Forest Regression")

    st.write(
        "An ensemble learning model that combines multiple "
        "decision trees to make predictions."
    )

    st.write("### 3. Gradient Boosting Regression")

    st.write(
        "A boosting technique that builds models sequentially "
        "to reduce prediction errors."
    )

    st.write("### 4. XGBoost Regression")

    st.write(
        "An optimized gradient boosting algorithm designed "
        "for efficient and accurate machine learning."
    )

    st.subheader("Evaluation Metrics")

    st.write("""
    **MAE:** Mean Absolute Error

    **MSE:** Mean Squared Error

    **RMSE:** Root Mean Squared Error

    **R²:** Coefficient of Determination
    """)

    st.info(
        "On the test split used in this project, the Random Forest "
        "model achieved the highest R² among the four tested models."
    )


# =========================================================
# 5. CHARTS
# =========================================================

elif page == "Charts":

    st.header("📊 Project Charts")

    # -----------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------

    st.subheader("Model Comparison - R² Score")

    chart_path = os.path.join(
        PLOTS_DIR,
        "model_comparison_r2.png"
    )

    if os.path.exists(chart_path):

        st.image(
            chart_path,
            use_container_width=True
        )

    else:

        st.warning(
            "Model comparison chart not found."
        )

    # -----------------------------------------------------
    # ACTUAL VS PREDICTED
    # -----------------------------------------------------

    st.subheader("Actual AQI vs Predicted AQI")

    chart_path = os.path.join(
        PLOTS_DIR,
        "actual_vs_predicted_aqi.png"
    )

    if os.path.exists(chart_path):

        st.image(
            chart_path,
            use_container_width=True
        )

    else:

        st.warning(
            "Actual vs predicted chart not found."
        )

    # -----------------------------------------------------
    # RESIDUAL PLOT
    # -----------------------------------------------------

    st.subheader("Residual Plot - Random Forest")

    chart_path = os.path.join(
        PLOTS_DIR,
        "residual_plot_rf.png"
    )

    if os.path.exists(chart_path):

        st.image(
            chart_path,
            use_container_width=True
        )

    else:

        st.warning(
            "Residual plot not found."
        )


# =========================================================
# 6. AQI PREDICTION
# =========================================================

elif page == "AQI Prediction":

    st.header("🌫️ Predict AQI")

    st.write(
        "Enter the air pollution parameters below to predict AQI."
    )

    # -----------------------------------------------------
    # CITY AND DATE
    # -----------------------------------------------------

    cities = sorted(
        df["City"].dropna().unique()
    )

    city = st.selectbox(
        "City",
        cities
    )

    date = st.date_input(
        "Date"
    )

    # -----------------------------------------------------
    # DEFAULT VALUES
    # -----------------------------------------------------

    numeric_columns = [
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NOx",
        "NH3",
        "CO",
        "SO2",
        "O3",
        "Benzene",
        "Toluene",
        "Xylene"
    ]

    defaults = {}

    for col in numeric_columns:

        defaults[col] = df[col].median()

    # -----------------------------------------------------
    # POLLUTANT INPUTS
    # -----------------------------------------------------

    st.subheader("Pollutant Values")

    col1, col2, col3 = st.columns(3)

    with col1:

        pm25 = st.number_input(
            "PM2.5",
            min_value=0.0,
            value=float(defaults["PM2.5"]),
            step=0.1
        )

        pm10 = st.number_input(
            "PM10",
            min_value=0.0,
            value=float(defaults["PM10"]),
            step=0.1
        )

        no = st.number_input(
            "NO",
            min_value=0.0,
            value=float(defaults["NO"]),
            step=0.1
        )

        no2 = st.number_input(
            "NO2",
            min_value=0.0,
            value=float(defaults["NO2"]),
            step=0.1
        )

    with col2:

        nox = st.number_input(
            "NOx",
            min_value=0.0,
            value=float(defaults["NOx"]),
            step=0.1
        )

        nh3 = st.number_input(
            "NH3",
            min_value=0.0,
            value=float(defaults["NH3"]),
            step=0.1
        )

        co = st.number_input(
            "CO",
            min_value=0.0,
            value=float(defaults["CO"]),
            step=0.1
        )

        so2 = st.number_input(
            "SO2",
            min_value=0.0,
            value=float(defaults["SO2"]),
            step=0.1
        )

    with col3:

        o3 = st.number_input(
            "O3",
            min_value=0.0,
            value=float(defaults["O3"]),
            step=0.1
        )

        benzene = st.number_input(
            "Benzene",
            min_value=0.0,
            value=float(defaults["Benzene"]),
            step=0.1
        )

        toluene = st.number_input(
            "Toluene",
            min_value=0.0,
            value=float(defaults["Toluene"]),
            step=0.1
        )

        xylene = st.number_input(
            "Xylene",
            min_value=0.0,
            value=float(defaults["Xylene"]),
            step=0.1
        )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if st.button("🔮 Predict AQI"):

        input_data = pd.DataFrame({
            "City": [city],
            "PM2.5": [pm25],
            "PM10": [pm10],
            "NO": [no],
            "NO2": [no2],
            "NOx": [nox],
            "NH3": [nh3],
            "CO": [co],
            "SO2": [so2],
            "O3": [o3],
            "Benzene": [benzene],
            "Toluene": [toluene],
            "Xylene": [xylene],
            "Year": [date.year],
            "Month": [date.month],
            "Day": [date.day],
            "DayOfWeek": [date.weekday()]
        })

        try:

            prediction = model.predict(input_data)[0]

            prediction = max(
                0,
                float(prediction)
            )

            st.success(
                f"Predicted AQI: {prediction:.2f}"
            )

            # -------------------------------------------------
            # AQI CATEGORY
            # -------------------------------------------------

            if prediction <= 50:

                category = "Good"

            elif prediction <= 100:

                category = "Satisfactory"

            elif prediction <= 200:

                category = "Moderate"

            elif prediction <= 300:

                category = "Poor"

            elif prediction <= 400:

                category = "Very Poor"

            else:

                category = "Severe"

            st.info(
                f"Predicted AQI Category: {category}"
            )

            st.caption(
                "The prediction is generated by the trained "
                "Random Forest regression model."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )