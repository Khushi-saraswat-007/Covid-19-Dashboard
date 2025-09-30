import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("COVID-19 Interactive Dashboard")

# Load dataset
df = pd.read_csv("Covid Data.csv")
df['DATE_DIED'] = pd.to_datetime(df['DATE_DIED'], errors='coerce')

# Sidebar filters
st.sidebar.header("Filters")
age_range = st.sidebar.slider("Select Age Range", 0, 120, (0, 120))
gender_options = st.sidebar.multiselect("Select Gender", options=['Male','Female'], default=['Male','Female'])
patient_type_options = st.sidebar.multiselect(
    "Select Patient Type", df['PATIENT_TYPE'].unique(), default=df['PATIENT_TYPE'].unique()
)
comorbidities = ['DIABETES', 'HYPERTENSION', 'OBESITY']
selected_comorbidities = st.sidebar.multiselect("Select Comorbidities", options=comorbidities, default=comorbidities)

# Map gender to numeric
gender_map = {'Male':0, 'Female':1}

# Filter dataframe
df_filtered = df[(df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])]
df_filtered = df_filtered[df_filtered['SEX'].isin([gender_map[g] for g in gender_options])]
df_filtered = df_filtered[df_filtered['PATIENT_TYPE'].isin(patient_type_options)]
for c in selected_comorbidities:
    df_filtered = df_filtered[df_filtered[c].notna()]

# Top metrics
total_patients = df_filtered.shape[0]
total_deaths = df_filtered['DATE_DIED'].notna().sum()
avg_age = df_filtered['AGE'].mean()
median_age = df_filtered['AGE'].median()
death_rate = (total_deaths / total_patients * 100) if total_patients>0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Patients", total_patients)
col2.metric("Total Deaths", total_deaths)
col3.metric("Average Age", f"{avg_age:.1f}")
col4.metric("Median Age", f"{median_age:.1f}")

st.metric("Death Rate (%)", f"{death_rate:.2f}")

# Tabs for sections
tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Health Features", "Death & Recovery", "Insights"])

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

    # Stacked bar for patient type vs gender
    st.subheader("Patient Type by Gender")
    patient_gender = pd.crosstab(df_filtered['PATIENT_TYPE'], df_filtered['SEX'])
    patient_gender.plot(kind='bar', stacked=True, figsize=(8,4), colormap='Set2')
    plt.xlabel("Patient Type")
    plt.ylabel("Count")
    plt.title("Stacked Bar: Patient Type by Gender")
    st.pyplot(plt.gcf())

# ===== Tab 2: Health Features =====
with tab2:
    st.subheader("Age vs Diabetes Condition")
    fig, ax = plt.subplots()
    sns.scatterplot(x="AGE", y="DIABETES", data=df_filtered, alpha=0.3, ax=ax)
    ax.set_title("Age vs Diabetes")
    st.pyplot(fig)

    st.subheader("Boxplot: Age by Patient Type")
    fig, ax = plt.subplots()
    sns.boxplot(x="PATIENT_TYPE", y="AGE", data=df_filtered, palette="Set2", ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap of Health Features")
    corr = df_filtered.select_dtypes('number').corr()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ===== Tab 3: Death & Recovery =====
with tab3:
    st.subheader("Death Trend Over Time")
    death_trend = df_filtered['DATE_DIED'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,4))
    death_trend.plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Deaths")
    ax.set_title("COVID-19 Deaths Over Time")
    st.pyplot(fig)

    st.subheader("Misleading vs Corrected Death Trend")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        death_trend.plot(ax=ax)
        ax.set_ylim(200, 500)
        ax.set_title("Misleading Death Trend")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        death_trend.plot(ax=ax)
        ax.set_ylim(0, death_trend.max())
        ax.set_title("Correct Death Trend")
        st.pyplot(fig)

# ===== Tab 4: Insights =====
with tab4:
    st.subheader("Filtered Data Download")
    st.download_button(
        label="Download Filtered Data",
        data=df_filtered.to_csv(index=False),
        file_name="filtered_covid_data.csv",
        mime="text/csv"
    )

    st.write("**Key Observations:**")
    st.write(f"- Total Patients: {total_patients}")
    st.write(f"- Total Deaths: {total_deaths}")
    st.write(f"- Death Rate: {death_rate:.2f}%")
    st.write(f"- Median Age: {median_age}")
    st.write("- Most common comorbidities in selected filters: " +
             ", ".join(selected_comorbidities))
