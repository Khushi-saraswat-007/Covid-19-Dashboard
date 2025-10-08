# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="🌍 COVID-19 Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- STYLE ----------------------
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #fceaff 0%, #e3f2fd 100%);
        }
        h1, h2, h3 {
            color: #4b0082;
        }
        .stMetric label {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.title("🌸 Global COVID-19 Data Dashboard")
st.markdown("#### A beautiful interactive visualization made by **Khushi Saraswat** ")

# ---------------------- LOAD DATA ----------------------
try:
    df = pd.read_csv("country_wise_latest.csv")
except FileNotFoundError:
    st.sidebar.warning("Default dataset not found. Please upload your CSV file.")
    uploaded = st.sidebar.file_uploader("📂 Upload the COVID dataset CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.stop()

# Clean column names
df.columns = [c.strip() for c in df.columns]

# ---------------------- SIDEBAR FILTERS ----------------------
st.sidebar.header("🔍 Filter Options")
regions = df["WHO Region"].dropna().unique().tolist()
selected_region = st.sidebar.multiselect("Select WHO Region(s):", regions, default=regions)

filtered_df = df[df["WHO Region"].isin(selected_region)]

# ---------------------- METRICS ROW ----------------------
st.subheader("🌏 Global Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Confirmed", f"{filtered_df['Confirmed'].sum():,}")
col2.metric("Total Deaths", f"{filtered_df['Deaths'].sum():,}")
col3.metric("Total Recovered", f"{filtered_df['Recovered'].sum():,}")
col4.metric("Total Active", f"{filtered_df['Active'].sum():,}")

st.markdown("---")

# ---------------------- BAR CHARTS ----------------------
st.subheader("📊 Top 15 Countries by Confirmed & Deaths")

colA, colB = st.columns(2)

with colA:
    top_confirmed = filtered_df.sort_values("Confirmed", ascending=False).head(15)
    fig1 = px.bar(
        top_confirmed,
        x="Confirmed", y="Country/Region",
        orientation="h",
        color="Confirmed",
        color_continuous_scale="Viridis",
        title="Top 15 Countries by Confirmed Cases"
    )
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    top_deaths = filtered_df.sort_values("Deaths", ascending=False).head(15)
    fig2 = px.bar(
        top_deaths,
        x="Deaths", y="Country/Region",
        orientation="h",
        color="Deaths",
        color_continuous_scale="Reds",
        title="Top 15 Countries by Deaths"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------------------- SCATTER PLOT ----------------------
st.subheader("🧭 Recovered vs Active Cases (Interactive Scatter)")

fig3 = px.scatter(
    filtered_df,
    x="Recovered",
    y="Active",
    color="WHO Region",
    size="Confirmed",
    hover_name="Country/Region",
    log_x=True,
    log_y=True,
    size_max=40,
    title="Recovered vs Active (Bubble size = Confirmed Cases)"
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ---------------------- CORRELATION HEATMAP ----------------------
st.subheader("🔗 Correlation Between COVID Metrics")
num_cols = ["Confirmed", "Deaths", "Recovered", "Active"]
corr = filtered_df[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f", linewidths=0.5)
st.pyplot(fig)

st.markdown("---")

# ---------------------- DISTRIBUTION ----------------------
st.subheader("📈 Distribution of Deaths per 100 Cases")
fig4 = px.histogram(
    filtered_df,
    x="Deaths / 100 Cases",
    nbins=30,
    color_discrete_sequence=["#E75480"],
    title="Distribution: Deaths / 100 Cases"
)
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ---------------------- REGION-WISE COMPARISON ----------------------
st.subheader("🌎 Average Cases & Deaths by WHO Region")

region_stats = filtered_df.groupby("WHO Region")[["Confirmed", "Deaths", "Recovered"]].mean().reset_index()
fig5 = px.bar(
    region_stats.melt(id_vars="WHO Region", var_name="Metric", value_name="Value"),
    x="WHO Region", y="Value", color="Metric",
    barmode="group",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title="Average Cases & Deaths by WHO Region"
)
st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ---------------------- DOWNLOAD ----------------------
st.sidebar.header("⬇️ Download Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download Filtered Data", csv, "filtered_covid_data.csv", "text/csv")

# ---------------------- FOOTER ----------------------
st.markdown("""
---
🎨 **Dashboard designed by Khushi Saraswat**  
💡 *Data visualisation using Streamlit, Seaborn, and Plotly*  
""")
