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
/* ---- General App Styling ---- */
body, .main, .block-container {
    background-color: white !important;
    color: black !important;
}

/* ---- Sidebar ---- */
.css-1d391kg, [data-testid="stSidebar"], .stSidebar, .sidebar-content {
    background-color: white !important;
    color: black !important;
}

/* ---- Headers ---- */
h1, h2, h3, h4, h5, h6 {
    color: black !important;
}

/* ---- Markdown, Labels, Paragraphs ---- */
p, span, div, label, .stMarkdown, .css-q8sbsg, .css-1cpxqw2 {
    color: black !important;
}

/* ---- Metric Labels ---- */
.stMetric label {
    font-size: 16px;
    color: black !important;
}

/* ---- Metric Values ---- */
[data-testid="stMetricValue"] {
    color: black !important;
}

/* ---- Sidebar Text ---- */
[data-testid="stSidebar"] * {
    color: black !important;
}

/* ---- Plotly Hover Text ---- */
.js-plotly-plot text, .plotly .hoverlayer text {
    fill: black !important;
}

/* ---- Input Widgets ---- */
input, select, textarea {
    color: black !important;
    background-color: white !important;
}
</style>
""", unsafe_allow_html=True)


#  HEADER 
st.title("🌸 Global COVID-19 Data Dashboard")
st.markdown("#### A beautiful interactive visualization made by **Khushi Saraswat**")

# LOAD DATA 
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

#  SIDEBAR FILTERS 
st.sidebar.header("🔍 Filter Options")
regions = df["WHO Region"].dropna().unique().tolist()
selected_region = st.sidebar.multiselect("Select WHO Region(s):", regions, default=regions)
filtered_df = df[df["WHO Region"].isin(selected_region)]

#  METRICS 
st.subheader("🌏 Global Overview")
col1, col2, col3, col4 = st.columns(4)

def safe_sum(series):
    return series.sum() if not series.empty else 0

col1.metric("Total Confirmed", f"{safe_sum(filtered_df['Confirmed']):,}")
col2.metric("Total Deaths", f"{safe_sum(filtered_df['Deaths']):,}")
col3.metric("Total Recovered", f"{safe_sum(filtered_df['Recovered']):,}")
col4.metric("Total Active", f"{safe_sum(filtered_df['Active']):,}")
st.markdown("---")

# BAR CHARTS 
st.subheader("📊 Top 15 Countries by Confirmed & Deaths")
colA, colB = st.columns(2)

# Top Confirmed
with colA:
    if not filtered_df.empty:
        top_confirmed = filtered_df.sort_values("Confirmed", ascending=False).head(15)
        fig1 = px.bar(
            top_confirmed,
            x="Confirmed",
            y="Country/Region",
            orientation="h",
            color="Confirmed",
            color_continuous_scale=px.colors.sequential.Viridis,
            text="Confirmed",
            title="Top 15 Countries by Confirmed Cases"
        )
        fig1.update_layout(
            xaxis_tickformat=",",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="black",
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("**Insight:** These countries have the highest confirmed COVID-19 cases globally.")
    else:
        st.info("No data available for selected region(s).")

# Top Deaths
with colB:
    if not filtered_df.empty:
        top_deaths = filtered_df.sort_values("Deaths", ascending=False).head(15)
        fig2 = px.bar(
            top_deaths,
            x="Deaths",
            y="Country/Region",
            orientation="h",
            color="Deaths",
            color_continuous_scale=px.colors.sequential.Reds,
            text="Deaths",
            title="Top 15 Countries by Deaths"
        )
        fig2.update_layout(
            xaxis_tickformat=",",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="black",
            xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
            yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**Insight:** These countries have suffered the most fatalities.")
    else:
        st.info("No data available for selected region(s).")

st.markdown("---")

# SCATTER PLOT 
st.subheader("🧭 Recovered vs Active Cases (Interactive Scatter)")
if not filtered_df.empty:
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
    fig3.update_layout(
        xaxis_tickformat=",",
        yaxis_tickformat=",",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="black",
        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("**Insight:** Log scale helps visualize wide-ranging numbers; bubble size shows confirmed cases.")
else:
    st.info("No data available for scatter plot.")

st.markdown("---")

#TREE MAP 
st.subheader("🌳 Cases Distribution by Region and Country")
if not filtered_df.empty:
    fig_tree = px.treemap(
        filtered_df,
        path=["WHO Region", "Country/Region"],
        values="Confirmed",
        color="Deaths",
        color_continuous_scale=px.colors.sequential.Reds,
        title="Tree Map: Confirmed Cases & Deaths"
    )
    fig_tree.update_layout(paper_bgcolor="white", font_color="black")
    st.plotly_chart(fig_tree, use_container_width=True)
    st.markdown("**Insight:** Visualizes region-wise contributions and severity by deaths.")
else:
    st.info("No data available for tree map.")

st.markdown("---")

# BOX PLOT 
st.subheader("📦 Distribution of Deaths by Region")
if not filtered_df.empty:
    fig_box = px.box(
        filtered_df,
        x="WHO Region",
        y="Deaths",
        color="WHO Region",
        title="Box Plot: Deaths by WHO Region"
    )
    fig_box.update_layout(
        paper_bgcolor="white",
        font_color="black",
        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown("**Insight:** Shows median, quartiles, and outliers for deaths by region.")
else:
    st.info("No data available for box plot.")

st.markdown("---")

# PIE CHART 
st.subheader("🥧 Share of Confirmed Cases by Region")
if not filtered_df.empty:
    region_share = filtered_df.groupby("WHO Region")["Confirmed"].sum().reset_index()
    fig_pie = px.pie(
        region_share,
        names="WHO Region",
        values="Confirmed",
        title="Pie Chart: Confirmed Cases Share by Region",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_pie.update_layout(paper_bgcolor="white", font_color="black")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("**Insight:** Highlights regions with majority of confirmed cases.")
else:
    st.info("No data available for pie chart.")

st.markdown("---")

#  CORRELATION HEATMAP 
st.subheader("🔗 Correlation Between COVID Metrics")
if not filtered_df.empty:
    num_cols = ["Confirmed", "Deaths", "Recovered", "Active"]
    corr = filtered_df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="mako", fmt=".2f", linewidths=0.5)
    fig.patch.set_facecolor('white')
    st.pyplot(fig)
    st.markdown("**Insight:** Shows strong positive correlation between confirmed, deaths, and recovered metrics.")
else:
    st.info("No data available for correlation heatmap.")

st.markdown("---")

# DISTRIBUTION 
st.subheader("📈 Distribution of Deaths per 100 Cases")
if not filtered_df.empty:
    fig4 = px.histogram(
        filtered_df,
        x="Deaths / 100 Cases",
        nbins=30,
        color_discrete_sequence=["#E75480"],
        title="Distribution: Deaths / 100 Cases"
    )
    fig4.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="black",
        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("**Insight:** Most countries cluster around low death rates, few outliers have high rates.")
else:
    st.info("No data available for distribution.")

st.markdown("---")

# ---------------------- REGION-WISE COMPARISON ----------------------
st.subheader("🌎 Average Cases & Deaths by WHO Region")
if not filtered_df.empty:
    region_stats = filtered_df.groupby("WHO Region")[["Confirmed", "Deaths", "Recovered"]].mean().reset_index()
    fig5 = px.bar(
        region_stats.melt(id_vars="WHO Region", var_name="Metric", value_name="Value"),
        x="WHO Region",
        y="Value",
        color="Metric",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Average Cases & Deaths by WHO Region"
    )
    fig5.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color="black",
        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black'))
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("**Insight:** Highlights differences in average cases and deaths across regions.")
else:
    st.info("No data available for regional comparison.")

st.markdown("---")

st.sidebar.header("⬇️ Download Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download Filtered Data", csv, "filtered_covid_data.csv", "text/csv")

st.markdown("""
---
🎨 **Dashboard designed by Khushi Saraswat**  
💡 *Data visualisation using Streamlit, Seaborn, and Plotly*
""")
