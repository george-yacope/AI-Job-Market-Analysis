import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import warnings
from sklearn.preprocessing import MultiLabelBinarizer

# --- 1. Page Config ---
st.set_page_config(page_title="AI Job Market Dashboard", layout="wide")
warnings.filterwarnings('ignore')

# --- 2. Global Aesthetics & Palette ---
PALETTE = ['#023047', '#219ebc', '#ffb703', '#8ecae6', '#fb8500']
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

# --- 3. Helper Plotting Functions (From Notebook) ---
def get_counts(df, col, top_n=None):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, 'count']
    if top_n:
        counts = counts.head(top_n)
    counts['percent'] = (counts['count'] / counts['count'].sum() * 100).round(1)
    return counts

def plot_bar_chart(df, col, title, top_n=15, orientation='v', color=None):
    data = get_counts(df, col, top_n)
    if orientation == 'h':
        x_val, y_val = 'count', col
        data = data.sort_values('count', ascending=True)
    else:
        x_val, y_val = col, 'count'
        
    fig = px.bar(data, x=x_val, y=y_val, title=title, text='percent',
                 color=color, orientation=orientation)
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(showlegend=False, height=500)
    return fig

def plot_pie_chart(df, col, title):
    data = get_counts(df, col)
    fig = px.pie(data, names=col, values='count', title=title, hole=0.4,
                 color_discrete_sequence=PALETTE)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_distribution(df, col, title, nbins=40):
    fig = px.histogram(df, x=col, nbins=nbins, title=title, marginal="box",
                       color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(bargap=0.1)
    return fig

def plot_grouped_box(df, x_col, y_col, group_col, title, top_n=10):
    top_items = df[x_col].value_counts().head(top_n).index
    filtered_df = df[df[x_col].isin(top_items)]
    fig = px.box(filtered_df, x=x_col, y=y_col, color=group_col,
                 title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(xaxis_tickangle=-45, height=600)
    return fig

def plot_grouped_histogram(df, x_col, group_col, title, top_n=10):
    top_items = df[x_col].value_counts().head(top_n).index
    filtered_df = df[df[x_col].isin(top_items)]
    fig = px.histogram(filtered_df, x=x_col, color=group_col, barmode='group',
                       title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(xaxis_tickangle=-45, yaxis_title="Count", height=500)
    return fig

def plot_box_comparison(df, cat_col, num_col, title, color=None):
    order = df.groupby(cat_col)[num_col].median().sort_values().index
    fig = px.box(df, x=cat_col, y=num_col, title=title, color=color,
                 category_orders={cat_col: order})
    return fig

def plot_heatmap(df, x_col, y_col, title):
    pivot_table = pd.crosstab(df[y_col], df[x_col])
    fig = px.imshow(pivot_table, title=title, aspect='auto',
                    labels=dict(x=x_col, y=y_col, color="Count"))
    return fig

def plot_outlier_analysis(df, col, title="Outlier Analysis"):
    s = df[col].dropna()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        row_heights=[0.3, 0.7], vertical_spacing=0.05,
                        specs=[[{"type": "box"}], [{"type": "xy"}]])
    
    fig.add_trace(go.Box(x=s, name='Box Plot', marker_color=PALETTE[0], boxpoints='outliers'), row=1, col=1)
    fig.add_trace(go.Histogram(x=s, nbinsx=50, name='Distribution', marker_color=PALETTE[1]), row=2, col=1)
    
    mean_val = s.mean()
    median_val = s.median()
    fig.add_vline(x=mean_val, line_dash='dash', line_color=PALETTE[4], annotation_text=f'Mean: {mean_val:,.0f}')
    fig.add_vline(x=median_val, line_dash='dot', line_color=PALETTE[2], annotation_text=f'Median: {median_val:,.0f}', annotation_position='top left')
    fig.update_layout(title=title, height=600, showlegend=False)
    return fig

def plot_choropleth_map(df, loc_col, title):
    country_counts = df[loc_col].value_counts().reset_index()
    country_counts.columns = ['country', 'count']
    fig = px.choropleth(country_counts, locations="country", locationmode="country names",
                        color="count", hover_name="country", color_continuous_scale='Blues',
                        title=title)
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'), height=500)
    return fig

def plot_time_trend(df, x_col, y_col, title, agg_func='count'):
    # Pre-defined orders for correct sorting
    orders_dict = {
        'month': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
        'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'quarter': ['Q1', 'Q2', 'Q3', 'Q4']
    }
    
    if agg_func == 'count':
        data = df[x_col].value_counts().reset_index()
        data.columns = [x_col, 'value']
        y_label = 'Number of Jobs'
    else:
        data = df.groupby(x_col)[y_col].mean().reset_index()
        data.columns = [x_col, 'value']
        y_label = 'Avg Salary'

    fig = px.bar(data, x=x_col, y='value', title=title, text_auto='.2s',
                 color=x_col, category_orders=orders_dict)
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_label, showlegend=False)
    return fig

# --- 4. Data Loading & Preprocessing ---
@st.cache_data
def load_data():
    with st.spinner('Loading data...'):
        df = pd.read_csv("data/raw_ai_job_market.csv")
        
        # Basic Cleaning
        df = df.drop_duplicates()
        if 'job_id' in df.columns:
            df = df.drop(columns=['job_id'])

        # Location
        df[['city', 'country_code']] = df['location'].str.split(',', n=1, expand=True).apply(lambda x: x.str.strip())
        
        # Fallback for mapping if file not found
        try:
            with open('data/state_map.json', 'r') as f:
                location_map = json.load(f)
            df['country_code'] = df['country_code'].apply(lambda x: x.split(', ')[-1] if ',' in x else x)
            df['country'] = df['country_code'].map(location_map).fillna(df['country_code'])
        except:
            df['country'] = df['country_code'] # Use raw code if map fails
            
        df = df.drop(columns=['location', 'country_code'])

        # Salary
        df[['min_salary', 'max_salary']] = df['salary_range_usd'].str.split('-', n=1, expand=True).astype(float)
        df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2
        df = df.drop(columns=['salary_range_usd'])

        # Dates
        df['posted_date'] = pd.to_datetime(df['posted_date'])
        df['year'] = df['posted_date'].dt.year
        df['month'] = df['posted_date'].dt.month_name()
        df['day_name'] = df['posted_date'].dt.day_name()
        df['quarter'] = 'Q' + df['posted_date'].dt.quarter.astype(str)

        # Skills (Keep separate list for easy filtering/counting)
        df['skills_list'] = df[['skills_required', 'tools_preferred']].apply(
            lambda x: [s.strip() for s in (str(x['skills_required']) + ',' + str(x['tools_preferred'])).split(',') if s.strip() and s != 'nan'], axis=1
        )
        df['skills_count'] = df['skills_list'].apply(len)

        # One-Hot Encoding for Analysis
        mlb = MultiLabelBinarizer()
        skills_encoded = mlb.fit_transform(df['skills_list'])
        skills_df = pd.DataFrame(skills_encoded, columns=mlb.classes_, index=df.index)
        df = pd.concat([df, skills_df], axis=1)
        
        # Clean original strings
        df.drop(columns=['skills_required', 'tools_preferred'], inplace=True)
        
        return df, mlb.classes_

# Load Data
try:
    df_original, skill_columns = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- 5. Sidebar & Filtering ---
st.sidebar.title("Data Filters")

# Country Filter
all_countries = ['All'] + sorted(df_original['country'].unique().tolist())
selected_country = st.sidebar.selectbox("Select Country", all_countries)

# Experience Filter
all_exp = ['All'] + sorted(df_original['experience_level'].unique().tolist())
selected_exp = st.sidebar.selectbox("Select Experience Level", all_exp)

# Job Title Filter
all_titles = ['All'] + sorted(df_original['job_title'].unique().tolist())
selected_title = st.sidebar.selectbox("Select Job Title", all_titles)

# Applying Filters
df = df_original.copy()
if selected_country != 'All':
    df = df[df['country'] == selected_country]
if selected_exp != 'All':
    df = df[df['experience_level'] == selected_exp]
if selected_title != 'All':
    df = df[df['job_title'] == selected_title]

# --- 6. Main Layout (KPIs) ---
st.title("AI Job Market Analysis")
st.markdown(f"Analyzing **{len(df)}** job postings based on current selection.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Jobs", len(df))
kpi2.metric("Avg Salary", f"${df['avg_salary'].mean():,.0f}")
kpi3.metric("Top Industry", df['industry'].mode()[0] if not df.empty else "N/A")
kpi4.metric("Avg Skills per Job", f"{df['skills_count'].mean():.1f}")

st.markdown("---")

# --- 7. Tabs Structure (Matching Notebook Sections) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Market Overview", 
    "2. Salary & Compensation", 
    "3. Location & Geography", 
    "4. Time Series", 
    "5. Skills Analysis"
])

# === Tab 1: Market Overview ===
with tab1:
    st.header("1. Market Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_bar_chart(df, 'company_name', 'Who is hiring the most?', top_n=10, orientation='v'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_bar_chart(df, 'industry', 'Which industries are dominating?', top_n=10, orientation='v'), use_container_width=True)
        
    st.plotly_chart(plot_grouped_histogram(df, 'job_title', 'experience_level', 'Common Roles by Experience Level'), use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_pie_chart(df, 'employment_type', 'Employment Types'), use_container_width=True)
    with col4:
        st.plotly_chart(plot_pie_chart(df, 'company_size', 'Company Sizes'), use_container_width=True)

# === Tab 2: Salary Analysis ===
with tab2:
    st.header("2. Salary & Compensation")
    
    st.plotly_chart(plot_distribution(df, 'avg_salary', 'Salary Distribution (USD)'), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_grouped_box(df, 'job_title', 'avg_salary', 'experience_level', 'Salary by Role & Experience'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_grouped_box(df, 'company_size', 'avg_salary', 'experience_level', 'Salary by Company Size & Experience'), use_container_width=True)
        
    st.plotly_chart(plot_box_comparison(df, 'employment_type', 'avg_salary', 'Salary vs Employment Type', color='employment_type'), use_container_width=True)
    st.plotly_chart(plot_outlier_analysis(df, 'avg_salary', 'Salary Outlier Analysis'), use_container_width=True)

# === Tab 3: Location Analysis ===
with tab3:
    st.header("3. Location & Geography")
    
    st.plotly_chart(plot_choropleth_map(df, 'country', 'Global AI Jobs Distribution'), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_bar_chart(df, 'country', 'Top 10 Countries by Job Count', top_n=10), use_container_width=True)
    with col2:
        # Top 10 countries for salary box plot
        top_c = df['country'].value_counts().head(10).index
        df_loc = df[df['country'].isin(top_c)]
        st.plotly_chart(plot_box_comparison(df_loc, 'country', 'avg_salary', 'Salary in Top 10 Countries'), use_container_width=True)
        
    # Heatmap
    top_ind = df['industry'].value_counts().head(10).index
    df_heat = df[df['country'].isin(top_c) & df['industry'].isin(top_ind)]
    st.plotly_chart(plot_heatmap(df_heat, 'industry', 'country', 'Industry Concentration by Country'), use_container_width=True)

# === Tab 4: Time Series ===
with tab4:
    st.header("4. Time Series Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_time_trend(df, 'quarter', None, 'Hiring Volume per Quarter', 'count'), use_container_width=True)
    with col2:
        st.plotly_chart(plot_time_trend(df, 'quarter', 'avg_salary', 'Avg Salary per Quarter', 'mean'), use_container_width=True)
        
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_time_trend(df, 'month', None, 'Hiring Volume per Month', 'count'), use_container_width=True)
    with col4:
        st.plotly_chart(plot_time_trend(df, 'month', 'avg_salary', 'Avg Salary per Month', 'mean'), use_container_width=True)
        
    st.plotly_chart(plot_time_trend(df, 'day_name', None, 'Hiring Volume per Day', 'count'), use_container_width=True)

# === Tab 5: Skills Analysis ===
with tab5:
    st.header("5. Skills Analysis")
    
    # Filter existing skills in current df columns
    target_skills = ['AWS', 'Azure', 'BigQuery', 'C++', 'CUDA', 'Excel', 'FastAPI', 'Flask', 'GCP', 
                     'Hugging Face', 'KDB+', 'Keras', 'LangChain', 'MLflow', 'NumPy', 'Pandas', 
                     'Power BI', 'PyTorch', 'Python', 'R', 'Reinforcement Learning', 'SQL', 
                     'Scikit-learn', 'TensorFlow']
    
    existing_skills = [col for col in target_skills if col in df.columns]
    
    if existing_skills:
        # 1. Standard Tech Stack
        skills_sum = df[existing_skills].sum().sort_values(ascending=False).reset_index()
        skills_sum.columns = ['Skill', 'Count']
        skills_sum['percent'] = (skills_sum['Count'] / len(df) * 100).round(1)
        
        fig_stack = px.bar(skills_sum.head(20), x='Count', y='Skill', orientation='h', 
                           title='Top 20 In-Demand Skills', text='percent', color='Count')
        fig_stack.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_stack.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, height=600)
        st.plotly_chart(fig_stack, use_container_width=True)
        
        # 2. Heatmaps
        col1, col2 = st.columns(2)
        with col1:
            # Skill vs Role
            job_skill_matrix = df.groupby('job_title')[existing_skills].sum()
            st.plotly_chart(px.imshow(job_skill_matrix, title='Skill Frequency per Job Title', aspect='auto', labels=dict(color="Count")), use_container_width=True)
        
        with col2:
            # Skill vs Company Size
            size_skill_matrix = df.groupby('company_size')[existing_skills].sum()
            st.plotly_chart(px.imshow(size_skill_matrix, title='Skill Demand by Company Size', aspect='auto', labels=dict(color="Count")), use_container_width=True)
            
        # 3. Complexity
        st.plotly_chart(plot_distribution(df, 'skills_count', 'Tech Stack Complexity (Skills per Job)'), use_container_width=True)
    else:
        st.warning("Not enough data to display skills analysis for the current filters.")

# Footer
st.markdown("---")

st.caption("Dashboard generated from AI Job Market Analysis")
