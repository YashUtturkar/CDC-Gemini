import os
import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="CDC Natality Dashboard 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLOR_PRIMARY = "#2E5B88"
COLOR_ACCENT = "#D9534F"
COLOR_FEMALE = "#8E44AD"
COLOR_MALE = "#2980B9"

STATE_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
@st.cache_data
def load_data(file_name: str = "Provisional_Natality_2025_CDC1.csv") -> pd.DataFrame:
    file_path = file_name
    if not os.path.exists(file_path):
        alt_path = os.path.join("data", file_name)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found at '{file_path}' or '{alt_path}'.")

    df = pd.read_csv(file_path)

    required_cols = {"state_of_residence", "month", "month_code", "year_code", "sex_of_infant", "births"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    df["births"] = pd.to_numeric(df["births"], errors="coerce")
    if df["births"].isnull().any():
        st.warning("Found null or non-numeric values in birth counts; dropping affected rows.")
        df = df.dropna(subset=["births"])

    df["births"] = df["births"].astype(int)
    df["state_abbr"] = df["state_of_residence"].map(STATE_TO_ABBR)

    month_order = (
        df[["month_code", "month"]]
        .drop_duplicates()
        .sort_values("month_code")["month"]
        .tolist()
    )
    df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

    return df

# ==========================================
# 3. FILTER HELPERS
# ==========================================
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df.copy()

    selected_states = st.session_state.get("selected_states", [])
    selected_months = st.session_state.get("selected_months", [])
    selected_sex = st.session_state.get("selected_sex", "All")

    if selected_states:
        filtered_df = filtered_df[filtered_df["state_of_residence"].isin(selected_states)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    if selected_months:
        filtered_df = filtered_df[filtered_df["month"].isin(selected_months)]
    else:
        filtered_df = filtered_df.iloc[0:0]

    if selected_sex != "All":
        filtered_df = filtered_df[filtered_df["sex_of_infant"] == selected_sex]

    return filtered_df

def build_filter_summary(all_states: list, all_months: list) -> str:
    selected_states = st.session_state.get("selected_states", [])
    selected_months = st.session_state.get("selected_months", [])
    selected_sex = st.session_state.get("selected_sex", "All")

    state_str = (
        "All States" if len(selected_states) == len(all_states) 
        else f"{len(selected_states)} of {len(all_states)} States Selected"
    )
    month_str = (
        "All Months" if len(selected_months) == len(all_months) 
        else f"{len(selected_months)} Months Selected"
    )

    return f"**Geography:** {state_str} | **Months:** {month_str} | **Sex:** {selected_sex}"

# ==========================================
# 4. PLOTLY VISUALIZERS
# ==========================================
def plot_monthly_trend(df: pd.DataFrame):
    trend_data = df.groupby("month", observed=True)["births"].sum().reset_index()
    fig = px.line(
        trend_data, x="month", y="births", markers=True,
        title="Monthly Birth Count Trend",
        labels={"month": "Month", "births": "Birth Count"}
    )
    fig.update_traces(line_color=COLOR_PRIMARY, line_width=3, marker_size=8,
                      hovertemplate="<b>%{x}</b><br>Births: %{y:,}<extra></extra>")
    fig.update_layout(yaxis=dict(rangemode="tozero", tickformat=","), hovermode="x unified")
    return fig

def plot_sex_comparison(df: pd.DataFrame):
    sex_data = df.groupby(["month", "sex_of_infant"], observed=True)["births"].sum().reset_index()
    fig = px.bar(
        sex_data, x="month", y="births", color="sex_of_infant", barmode="group",
        title="Monthly Birth Counts by Infant Sex",
        labels={"month": "Month", "births": "Birth Count", "sex_of_infant": "Infant Sex"},
        color_discrete_map={"Female": COLOR_FEMALE, "Male": COLOR_MALE}
    )
    fig.update_traces(hovertemplate="<b>%{x}</b> (%{fullData.name})<br>Births: %{y:,}<extra></extra>")
    fig.update_layout(yaxis=dict(rangemode="tozero", tickformat=","))
    return fig

def plot_state_ranking(df: pd.DataFrame, n_top: int = 15):
    ranking = df.groupby("state_of_residence")["births"].sum().reset_index().sort_values("births", ascending=True)
    fig = px.bar(
        ranking.tail(n_top), x="births", y="state_of_residence", orientation="h",
        title=f"Top Geographies by Selected Birth Count",
        labels={"state_of_residence": "State", "births": "Birth Count"},
        color_discrete_sequence=[COLOR_PRIMARY]
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Births: %{x:,}<extra></extra>")
    fig.update_layout(xaxis=dict(rangemode="tozero", tickformat=","))
    return fig

def plot_us_choropleth(df: pd.DataFrame):
    map_data = df.groupby(["state_abbr", "state_of_residence"])["births"].sum().reset_index()
    fig = px.choropleth(
        map_data, locations="state_abbr", locationmode="USA-states", color="births", scope="usa",
        color_continuous_scale="Viridis", title="US Map of Birth Counts by State",
        labels={"births": "Births", "state_abbr": "State"}, hover_name="state_of_residence"
    )
    fig.update_traces(hovertemplate="<b>%{hovertext}</b> (%{location})<br>Birth Count: %{z:,}<extra></extra>")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig

def plot_state_month_heatmap(df: pd.DataFrame):
    pivot_df = df.pivot_table(
        index="state_of_residence", columns="month", values="births", aggfunc="sum", observed=True
    ).fillna(0)
    fig = px.imshow(
        pivot_df, labels=dict(x="Month", y="State", color="Births"),
        x=pivot_df.columns.tolist(), y=pivot_df.index.tolist(),
        color_continuous_scale="Cividis", aspect="auto", title="State vs. Month Birth Count Intensity"
    )
    fig.update_traces(hovertemplate="State: <b>%{y}</b><br>Month: <b>%{x}</b><br>Births: %{z:,}<extra></extra>")
    return fig

def plot_top_bottom_comparison(df: pd.DataFrame, top_n: int = 5):
    geo_totals = df.groupby("state_of_residence")["births"].sum().reset_index().sort_values("births", ascending=False)
    if len(geo_totals) < (top_n * 2):
        top_states = geo_totals
    else:
        top_states = pd.concat([geo_totals.head(top_n), geo_totals.tail(top_n)])

    top_states = top_states.sort_values("births", ascending=True)

    fig = px.bar(
        top_states, x="births", y="state_of_residence", orientation="h",
        title=f"Comparison: Top {top_n} & Bottom {top_n} Selected Geographies",
        labels={"state_of_residence": "State", "births": "Birth Count"},
        color="births", color_continuous_scale="Tealgrn"
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Births: %{x:,}<extra></extra>")
    fig.update_layout(xaxis=dict(rangemode="tozero", tickformat=","))
    return fig

# ==========================================
# 5. HEADER & SIDEBAR COMPONENTS
# ==========================================
def render_header():
    st.title("CDC Provisional Natality Dashboard (2025)")
    st.caption("Designed for Undergraduate Business Analytics Students | Data Exploration Tool")
    
    st.info(
        "**Data Source Attribution & Disclaimers:**\n"
        "- **Source:** CDC National Center for Health Statistics (NCHS) Provisional Natality Dataset (2025).\n"
        "- **Notice:** Figures presented in this dashboard are **PROVISIONAL** and subject to future revisions.\n"
        "- **Metric Definition:** All values represent **absolute birth counts**, NOT birth rates. "
        "Direct comparisons between states do not reflect population-adjusted birth rates."
    )
    st.divider()

def render_sidebar(df):
    st.sidebar.header("Filter Controls")

    all_states = sorted(df["state_of_residence"].unique().tolist())
    all_months = df[["month_code", "month"]].drop_duplicates().sort_values("month_code")["month"].tolist()
    sex_options = ["All", "Female", "Male"]

    if "selected_states" not in st.session_state:
        st.session_state.selected_states = all_states
    if "selected_months" not in st.session_state:
        st.session_state.selected_months = all_months
    if "selected_sex" not in st.session_state:
        st.session_state.selected_sex = "All"

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Select All", use_container_width=True):
        st.session_state.selected_states = all_states
        st.session_state.selected_months = all_months
        st.session_state.selected_sex = "All"
        st.rerun()

    if col2.button("Reset Filters", use_container_width=True):
        st.session_state.selected_states = all_states
        st.session_state.selected_months = all_months
        st.session_state.selected_sex = "All"
        st.rerun()

    st.sidebar.divider()

    st.session_state.selected_states = st.sidebar.multiselect(
        "Select State/Geography:",
        options=all_states,
        default=st.session_state.selected_states
    )

    st.session_state.selected_months = st.sidebar.multiselect(
        "Select Months:",
        options=all_months,
        default=st.session_state.selected_months
    )

    st.session_state.selected_sex = st.sidebar.radio(
        "Infant Sex Filter:",
        options=sex_options,
        index=sex_options.index(st.session_state.selected_sex)
    )

    return all_states, all_months

def render_kpis(filtered_df: pd.DataFrame):
    if filtered_df.empty:
        st.warning("No data matches current filter selection. Adjust sidebar filters.")
        return

    total_births = filtered_df["births"].sum()
    selected_geos = filtered_df["state_of_residence"].nunique()
    
    num_months = filtered_df["month"].nunique()
    avg_monthly_births = (total_births / num_months) if num_months > 0 else 0

    geo_totals = filtered_df.groupby("state_of_residence")["births"].sum()
    top_geo = geo_totals.idxmax() if not geo_totals.empty else "N/A"
    top_geo_val = geo_totals.max() if not geo_totals.empty else 0

    month_totals = filtered_df.groupby("month", observed=True)["births"].sum()
    top_month = month_totals.idxmax() if not month_totals.empty else "N/A"
    top_month_val = month_totals.max() if not month_totals.empty else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Births", f"{total_births:,}")
    with col2:
        st.metric("Geographies", f"{selected_geos}")
    with col3:
        st.metric("Avg Births / Month", f"{int(round(avg_monthly_births)):,}")
    with col4:
        st.metric("Top Geography", f"{top_geo}", f"{top_geo_val:,} births")
    with col5:
        st.metric("Peak Month", f"{top_month}", f"{top_month_val:,} births")

    st.divider()

# ==========================================
# 6. MAIN APPLICATION LOGIC
# ==========================================
def main():
    render_header()

    try:
        raw_df = load_data("Provisional_Natality_2025_CDC1.csv")
    except Exception as e:
        st.error(f"Error loading natality dataset: {e}")
        st.stop()

    all_states, all_months = render_sidebar(raw_df)
    filtered_df = apply_filters(raw_df)

    st.markdown(build_filter_summary(all_states, all_months))
    st.write("")

    render_kpis(filtered_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Geographic Analysis",
        "Monthly & Sex Analysis",
        "Data Table & Download",
        "About the Data"
    ])

    # TAB 1: OVERVIEW
    with tab1:
        if filtered_df.empty:
            st.warning("No data available for the active selections.")
        else:
            st.subheader("Overview & Key Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_monthly_trend(filtered_df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_top_bottom_comparison(filtered_df, top_n=5), use_container_width=True)

    # TAB 2: GEOGRAPHIC ANALYSIS
    with tab2:
        if filtered_df.empty:
            st.warning("No data available for geographic analysis.")
        else:
            st.subheader("Geographic Distribution")
            col1, col2 = st.columns([1.2, 1.0])
            with col1:
                st.plotly_chart(plot_us_choropleth(filtered_df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_state_ranking(filtered_df, n_top=15), use_container_width=True)

    # TAB 3: MONTHLY & SEX ANALYSIS
    with tab3:
        if filtered_df.empty:
            st.warning("No data available for monthly and sex analysis.")
        else:
            st.subheader("Monthly Seasonality & Sex Distribution")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_sex_comparison(filtered_df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_state_month_heatmap(filtered_df), use_container_width=True)

    # TAB 4: DATA TABLE & DOWNLOAD
    with tab4:
        st.subheader("Filtered Natality Dataset")
        if filtered_df.empty:
            st.warning("Current filter selection yields 0 records.")
        else:
            display_df = filtered_df[["state_of_residence", "month", "sex_of_infant", "births"]].copy()
            display_df.columns = ["State", "Month", "Infant Sex", "Birth Count"]

            st.dataframe(
                display_df.style.format({"Birth Count": "{:,}"}),
                use_container_width=True,
                hide_index=True
            )

            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Filtered Data (CSV)",
                data=csv_data,
                file_name="filtered_cdc_natality_2025.csv",
                mime="text/csv",
                use_container_width=False
            )

    # TAB 5: ABOUT THE DATA
    with tab5:
        st.subheader("About the CDC Provisional Natality Dataset")
        st.markdown("""
        ### Data Source & Methodology
        This application utilizes provisional birth count records supplied by the **Centers for Disease Control and Prevention (CDC) National Center for Health Statistics (NCHS)**.
        
        ### Key Analytical Considerations for Business Analytics Students:
        1. **Birth Counts vs. Birth Rates**:
           - **Counts**: The absolute total number of live births recorded in a jurisdiction.
           - **Rates**: Births per unit of population (e.g., births per 1,000 live female residents aged 15–44).
           - *Caution*: High population states (e.g., California, Texas) naturally produce higher birth counts than lower population states (e.g., Wyoming) regardless of demographic fertility trends.
        2. **Provisional Status**:
           - Provisional records are based on flow data received by NCHS and are incomplete. Final annual natality reports may adjust totals.
        3. **Seasonality & Sex Ratios**:
           - Human natality exhibits natural monthly seasonality (often peaking in late summer/early autumn) and consistent male-to-female sex ratios at birth (~105 males per 100 females).
        """)

if __name__ == "__main__":
    main()
