import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("COVID-19 Data Dashboard")

# Load dataset directly
df = pd.read_csv("Covid Data.csv")  # Make sure the CSV is in the same folder
df['DATE_DIED'] = pd.to_datetime(df['DATE_DIED'], errors='coerce')

# ===== Top Info Cards =====
total_patients = df.shape[0]
total_deaths = df['DATE_DIED'].notna().sum()
avg_age = df['AGE'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Patients", total_patients)
col2.metric("Total Deaths", total_deaths)
col3.metric("Average Age", f"{avg_age:.1f}")

# ===== First Row: Age & Gender Distribution =====
st.subheader("Age & Gender Distribution")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.hist(df['AGE'], bins=30, color="skyblue")
    ax.set_title("Age Distribution")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.countplot(x="SEX", data=df, palette="viridis", ax=ax)
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Male","Female"])
    ax.set_title("Gender Distribution")
    st.pyplot(fig)

# ===== Second Row: Death Trend & Age vs Diabetes =====
st.subheader("Death Trend & Age vs Diabetes")
col1, col2 = st.columns(2)

with col1:
    death_trend = df['DATE_DIED'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6,4))
    death_trend.plot(ax=ax)
    ax.set_title("COVID-19 Deaths Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Deaths")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(x="AGE", y="DIABETES", data=df, alpha=0.3, ax=ax)
    ax.set_title("Age vs Diabetes")
    st.pyplot(fig)

# ===== Third Row: Boxplot & Correlation Heatmap =====
st.subheader("Patient Type & Feature Correlations")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.boxplot(x="PATIENT_TYPE", y="AGE", data=df, palette="Set2", ax=ax)
    ax.set_title("Age by Patient Type")
    st.pyplot(fig)

with col2:
    corr = df.select_dtypes('number').corr()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

# ===== Fourth Row: Misleading vs Corrected Death Trend =====
st.subheader("Misleading vs Corrected Death Trend")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    death_trend.plot(ax=ax)
    ax.set_ylim(200, 500)  # WRONG
    ax.set_title("Misleading Death Trend")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    death_trend.plot(ax=ax)
    ax.set_ylim(0, death_trend.max())
    ax.set_title("Correct Death Trend")
    st.pyplot(fig)
