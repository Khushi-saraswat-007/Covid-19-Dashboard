import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("Covid Data.csv")  # Make sure CSV is in the same folder
df['DATE_DIED'] = pd.to_datetime(df['DATE_DIED'], errors='coerce')

# Sidebar filters
st.sidebar.header("Filters")
age_range = st.sidebar.slider("Select Age Range", 0, 120, (0, 120))
gender_options = st.sidebar.multiselect("Select Gender", options=['Male','Female'], default=['Male','Female'])
patient_type_options = st.sidebar.multiselect("Select Patient Type", df['PATIENT_TYPE'].unique(), default=df['PATIENT_TYPE'].unique())

# Filter dataframe
df_filtered = df[(df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])]
df_filtered = df_filtered[df_filtered['SEX'].isin([0 if g=="Male" else 1 for g in gender_options])]
df_filtered = df_filtered[df_filtered['PATIENT_TYPE'].isin(patient_type_options)]

# Top metrics
st.title("COVID-19 Data Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", df_filtered.shape[0])
col2.metric("Total Deaths", df_filtered['DATE_DIED'].notna().sum())
col3.metric("Average Age", f"{df_filtered['AGE'].mean():.1f}")

# Tabs for sections
tab1, tab2, tab3 = st.tabs(["Demographics", "Health Features", "Death Analysis"])

# ===== Tab 1: Demographics =====
with tab1:
    st.subheader("Age & Gender Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(df_filtered['AGE'], bins=30, color="skyblue")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.set_title("Age Distribution")
        st.pyplot(fig)
        st.write(f"Most patients are around age {df_filtered['AGE'].mode()[0]}.")

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x="SEX", data=df_filtered, palette="viridis", ax=ax)
        ax.set_xticks([0,1])
        ax.set_xticklabels(["Male","Female"])
        ax.set_title("Gender Distribution")
        st.pyplot(fig)
        st.write(f"Gender distribution: {df_filtered['SEX'].value_counts().to_dict()} (0=Male, 1=Female)")

# ===== Tab 2: Health Features =====
with tab2:
    st.subheader("Age vs Diabetes & Patient Type")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.scatterplot(x="AGE", y="DIABETES", data=df_filtered, alpha=0.3, ax=ax)
        ax.set_title("Age vs Diabetes Condition")
        st.pyplot(fig)
        st.write("Scatter plot shows relation between patient age and diabetes occurrence.")

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x="PATIENT_TYPE", y="AGE", data=df_filtered, palette="Set2", ax=ax)
        ax.set_title("Age by Patient Type")
        st.pyplot(fig)
        st.write("Boxplot shows age distribution across patient types.")

    st.subheader("Correlation Heatmap of Health Features")
    corr = df_filtered.select_dtypes('number').corr()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    st.write("Heatmap shows correlation between numeric health features.")

# ===== Tab 3: Death Analysis =====
with tab3:
    st.subheader("Death Trend Over Time")
    death_trend = df_filtered['DATE_DIED'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,4))
    death_trend.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Deaths")
    ax.set_title("COVID-19 Deaths Over Time")
    st.pyplot(fig)
    st.write("Line chart shows the number of deaths over time.")

    st.subheader("Misleading vs Corrected Death Trend")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        death_trend.plot(ax=ax)
        ax.set_ylim(200, 500)  # Misleading
        ax.set_title("Misleading Death Trend")
        st.pyplot(fig)
        st.write("Truncated y-axis exaggerates trends.")

    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        death_trend.plot(ax=ax)
        ax.set_ylim(0, death_trend.max())  # Corrected
        ax.set_title("Correct Death Trend")
        st.pyplot(fig)
        st.write("Corrected y-axis shows the true trend of deaths.")
