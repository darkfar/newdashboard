import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Bike Sharing Analytics Dashboard",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visual design
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #3498db;
        padding-left: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .advanced-analysis {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the bike sharing data with enhanced features"""
    try:
        day_df = pd.read_csv('day.csv')
        hour_df = pd.read_csv('hour.csv')
    except FileNotFoundError:
        st.warning("Data files not found. Using enhanced sample data for demonstration.")
        # Generate more realistic sample data
        np.random.seed(42)
        dates = pd.date_range('2011-01-01', '2012-12-31', freq='D')
        
        # Create more realistic seasonal patterns
        day_df = pd.DataFrame({
            'dteday': dates,
            'season': [(d.month-1)//3 + 1 for d in dates],
            'yr': [d.year - 2011 for d in dates],
            'mnth': [d.month for d in dates],
            'holiday': np.random.choice([0, 1], len(dates), p=[0.97, 0.03]),
            'weekday': [d.weekday() for d in dates],
            'workingday': [(d.weekday() < 5 and np.random.random() > 0.03) for d in dates],
            'weathersit': np.random.choice([1, 2, 3, 4], len(dates), p=[0.6, 0.3, 0.09, 0.01]),
        })
        
        # Add realistic temperature patterns based on season and month
        for i, row in day_df.iterrows():
            month = row['mnth']
            # Temperature varies by month (normalized 0-1)
            base_temp = 0.3 + 0.4 * np.sin(2 * np.pi * (month - 1) / 12)
            day_df.at[i, 'temp'] = np.clip(base_temp + np.random.normal(0, 0.1), 0.05, 0.95)
            day_df.at[i, 'atemp'] = np.clip(day_df.at[i, 'temp'] + np.random.normal(0, 0.05), 0.05, 0.95)
            day_df.at[i, 'hum'] = np.random.uniform(0.3, 0.9)
            day_df.at[i, 'windspeed'] = np.random.uniform(0.0, 0.5)
        
        # Generate realistic usage patterns
        for i, row in day_df.iterrows():
            season_multiplier = {1: 0.7, 2: 1.2, 3: 1.3, 4: 0.8}[row['season']]
            weather_multiplier = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.2}[row['weathersit']]
            temp_multiplier = 0.5 + row['temp']
            weekend_multiplier = 0.8 if row['weekday'] >= 5 else 1.0
            
            base_usage = 4000 * season_multiplier * weather_multiplier * temp_multiplier * weekend_multiplier
            day_df.at[i, 'casual'] = int(base_usage * np.random.uniform(0.15, 0.35))
            day_df.at[i, 'registered'] = int(base_usage * np.random.uniform(0.65, 0.85))
            day_df.at[i, 'cnt'] = day_df.at[i, 'casual'] + day_df.at[i, 'registered']
        
        # Generate enhanced hourly data
        hour_data = []
        for _, day in day_df.iterrows():
            for hour in range(24):
                # More realistic hourly patterns
                if day['weekday'] < 5:  # Weekday
                    if hour in [7, 8]:  # Morning rush
                        hourly_multiplier = np.random.uniform(2.5, 3.5)
                    elif hour in [17, 18, 19]:  # Evening rush
                        hourly_multiplier = np.random.uniform(3.0, 4.0)
                    elif hour in [9, 10, 11, 12, 13, 14, 15, 16]:  # Work hours
                        hourly_multiplier = np.random.uniform(1.0, 1.5)
                    elif hour in [0, 1, 2, 3, 4, 5]:  # Night
                        hourly_multiplier = np.random.uniform(0.05, 0.15)
                    else:  # Evening/night
                        hourly_multiplier = np.random.uniform(0.5, 1.0)
                else:  # Weekend
                    if hour in [10, 11, 12, 13, 14, 15, 16]:  # Afternoon
                        hourly_multiplier = np.random.uniform(1.5, 2.5)
                    elif hour in [0, 1, 2, 3, 4, 5]:  # Night
                        hourly_multiplier = np.random.uniform(0.05, 0.15)
                    else:
                        hourly_multiplier = np.random.uniform(0.8, 1.2)
                
                base_hourly = day['cnt'] / 24
                hourly_cnt = int(base_hourly * hourly_multiplier)
                
                hour_data.append({
                    'dteday': day['dteday'],
                    'hr': hour,
                    'season': day['season'],
                    'yr': day['yr'],
                    'mnth': day['mnth'],
                    'holiday': day['holiday'],
                    'weekday': day['weekday'],
                    'workingday': day['workingday'],
                    'weathersit': day['weathersit'],
                    'temp': day['temp'],
                    'atemp': day['atemp'],
                    'hum': day['hum'],
                    'windspeed': day['windspeed'],
                    'casual': max(1, int(hourly_cnt * (0.3 if day['weekday'] >= 5 else 0.2))),
                    'registered': max(1, int(hourly_cnt * (0.7 if day['weekday'] >= 5 else 0.8))),
                    'cnt': max(2, hourly_cnt)
                })
        
        hour_df = pd.DataFrame(hour_data)
    
    # Enhanced preprocessing
    def preprocess_data(df):
        df = df.copy()
        df['dteday'] = pd.to_datetime(df['dteday'])
        df['year'] = df['dteday'].dt.year
        df['month'] = df['dteday'].dt.month
        df['day'] = df['dteday'].dt.day
        df['day_of_week'] = df['dteday'].dt.dayofweek
        df['week_of_year'] = df['dteday'].dt.isocalendar().week
        df['quarter'] = df['dteday'].dt.quarter
        return df
    
    day_df = preprocess_data(day_df)
    hour_df = preprocess_data(hour_df)
    
    # Enhanced mappings
    season_dict = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    weathersit_dict = {1: 'Clear/Partly Cloudy', 2: 'Misty/Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow/Fog'}
    month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                  7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    weekday_dict = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    
    # Apply mappings
    for df in [day_df, hour_df]:
        df['season_name'] = df['season'].map(season_dict)
        df['weathersit_name'] = df['weathersit'].map(weathersit_dict)
        df['month_name'] = df['month'].map(month_dict)
        df['weekday_name'] = df['day_of_week'].map(weekday_dict)
        
        # Additional categorical features
        df['is_weekend'] = df['weekday'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        df['temp_category'] = pd.cut(df['temp'], bins=[0, 0.33, 0.66, 1.0], labels=['Cold', 'Moderate', 'Hot'])
        df['usage_intensity'] = pd.cut(df['cnt'], bins=3, labels=['Low', 'Medium', 'High'])
    
    return day_df, hour_df

def create_robust_bins(data, n_bins=5, labels=None):
    """Create robust quantile bins that handle duplicate values"""
    if labels is None:
        labels = list(range(1, n_bins + 1))
    
    # Check if we have enough unique values
    unique_values = data.nunique()
    
    if unique_values < n_bins:
        # If not enough unique values, use simpler binning
        if unique_values <= 2:
            # For very few unique values, just use binary classification
            median_val = data.median()
            return pd.Series([labels[-1] if x >= median_val else labels[0] for x in data], 
                           index=data.index)
        else:
            # Use the number of unique values available
            n_bins = unique_values
            labels = labels[:n_bins]
    
    try:
        return pd.qcut(data, n_bins, labels=labels, duplicates='drop')
    except ValueError:
        # If qcut still fails, use regular cut
        return pd.cut(data, n_bins, labels=labels)

def perform_rfm_analysis(day_df):
    """
    Perform RFM Analysis for bike sharing data with robust error handling
    """
    st.markdown('<h3 class="advanced-analysis">🔍 RFM Analysis - User Behavior Segmentation</h3>', unsafe_allow_html=True)
    
    try:
        # Define high usage threshold (top 30% of daily usage)
        high_usage_threshold = day_df['cnt'].quantile(0.7)
        
        # Calculate RFM metrics by month (treating each month as a "customer")
        day_df['year_month'] = day_df['dteday'].dt.to_period('M')
        
        rfm_data = []
        for period in day_df['year_month'].unique():
            month_data = day_df[day_df['year_month'] == period]
            
            # Recency: Days since last high usage day in the month
            high_usage_days = month_data[month_data['cnt'] >= high_usage_threshold]
            if not high_usage_days.empty:
                last_high_usage = high_usage_days['dteday'].max()
                recency = (month_data['dteday'].max() - last_high_usage).days
            else:
                recency = 30  # Max recency if no high usage
            
            # Frequency: Number of high usage days in the month
            frequency = len(high_usage_days)
            
            # Monetary: Total usage in the month
            monetary = month_data['cnt'].sum()
            
            rfm_data.append({
                'period': str(period),
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary
            })
        
        rfm_df = pd.DataFrame(rfm_data)
        
        if len(rfm_df) < 3:
            st.warning("Not enough data points for RFM analysis. Need at least 3 months of data.")
            return rfm_df
        
        # Create RFM scores (1-5 scale) with robust binning
        rfm_df['R_score'] = create_robust_bins(rfm_df['recency'], n_bins=5, labels=[5,4,3,2,1])
        rfm_df['F_score'] = create_robust_bins(rfm_df['frequency'], n_bins=5, labels=[1,2,3,4,5])
        rfm_df['M_score'] = create_robust_bins(rfm_df['monetary'], n_bins=5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        rfm_df['R_score'] = pd.to_numeric(rfm_df['R_score'], errors='coerce').fillna(3)
        rfm_df['F_score'] = pd.to_numeric(rfm_df['F_score'], errors='coerce').fillna(3)
        rfm_df['M_score'] = pd.to_numeric(rfm_df['M_score'], errors='coerce').fillna(3)
        
        # Create RFM segments
        def create_rfm_segment(row):
            try:
                if row['R_score'] >= 4 and row['F_score'] >= 4 and row['M_score'] >= 4:
                    return 'Champions'
                elif row['R_score'] >= 3 and row['F_score'] >= 3 and row['M_score'] >= 3:
                    return 'Loyal Customers'
                elif row['R_score'] >= 3 and row['F_score'] <= 2:
                    return 'Potential Loyalists'
                elif row['R_score'] <= 2 and row['F_score'] >= 3:
                    return 'At Risk'
                elif row['R_score'] <= 2 and row['F_score'] <= 2 and row['M_score'] >= 3:
                    return 'Cannot Lose Them'
                else:
                    return 'Others'
            except:
                return 'Others'
        
        rfm_df['segment'] = rfm_df.apply(create_rfm_segment, axis=1)
        
        # Visualize RFM Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # RFM Distribution
            fig_rfm_dist = px.scatter_3d(
                rfm_df,
                x='recency',
                y='frequency',
                z='monetary',
                color='segment',
                title="RFM 3D Scatter Plot",
                labels={'recency': 'Recency (days)', 'frequency': 'Frequency', 'monetary': 'Monetary (total usage)'}
            )
            st.plotly_chart(fig_rfm_dist, use_container_width=True)
        
        with col2:
            # Segment distribution
            segment_counts = rfm_df['segment'].value_counts()
            fig_segments = px.pie(
                values=segment_counts.values,
                names=segment_counts.index,
                title="RFM Segment Distribution"
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        
        # RFM Summary Table
        st.subheader("RFM Segment Summary")
        segment_summary = rfm_df.groupby('segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        }).round(2)
        
        st.dataframe(segment_summary, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in RFM Analysis: {str(e)}")
        st.info("RFM Analysis requires sufficient data diversity. Using simplified analysis.")
        
        # Simplified analysis as fallback
        simplified_analysis = day_df.groupby('month').agg({
            'cnt': ['mean', 'sum', 'std']
        }).round(2)
        simplified_analysis.columns = ['Avg_Daily_Usage', 'Total_Monthly_Usage', 'Usage_Variability']
        
        st.subheader("Simplified Monthly Performance Analysis")
        st.dataframe(simplified_analysis, use_container_width=True)
        
        return pd.DataFrame({'period': ['fallback'], 'segment': ['Standard']})
    
    return rfm_df

def perform_clustering_analysis(day_df):
    """
    Perform Manual Clustering Analysis using business rules
    """
    st.markdown('<h3 class="advanced-analysis">📊 Manual Clustering Analysis - Usage Patterns</h3>', unsafe_allow_html=True)
    
    try:
        # Define clustering based on usage patterns and conditions
        def assign_cluster(row):
            usage = row['cnt']
            temp = row['temp']
            weather = row['weathersit']
            is_weekend = row['weekday'] >= 5
            
            # High usage, good conditions
            if usage >= day_df['cnt'].quantile(0.8) and weather <= 2 and temp >= 0.5:
                return 'Peak Performance'
            # Low usage, bad conditions
            elif usage <= day_df['cnt'].quantile(0.2) and (weather >= 3 or temp <= 0.3):
                return 'Weather Affected'
            # Weekend patterns
            elif is_weekend and usage >= day_df['cnt'].quantile(0.6):
                return 'Weekend Warriors'
            # Weekday moderate
            elif not is_weekend and day_df['cnt'].quantile(0.3) <= usage <= day_df['cnt'].quantile(0.7):
                return 'Regular Commuters'
            # High temp, moderate usage
            elif temp >= 0.7 and day_df['cnt'].quantile(0.4) <= usage <= day_df['cnt'].quantile(0.8):
                return 'Hot Weather Users'
            else:
                return 'Standard Usage'
        
        day_df['usage_cluster'] = day_df.apply(assign_cluster, axis=1)
        
        # Visualize clusters
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution over time
            cluster_time = day_df.groupby(['month_name', 'usage_cluster']).size().unstack(fill_value=0)
            
            # Create stacked bar chart
            colors = px.colors.qualitative.Set3
            fig_cluster_time = go.Figure()
            for i, cluster in enumerate(cluster_time.columns):
                fig_cluster_time.add_trace(go.Bar(
                    x=cluster_time.index,
                    y=cluster_time[cluster],
                    name=cluster,
                    marker_color=colors[i % len(colors)]
                ))
            
            fig_cluster_time.update_layout(
                title="Usage Clusters by Month",
                xaxis_title="Month",
                yaxis_title="Number of Days",
                barmode='stack'
            )
            st.plotly_chart(fig_cluster_time, use_container_width=True)
        
        with col2:
            # Cluster characteristics
            cluster_stats = day_df.groupby('usage_cluster').agg({
                'cnt': 'mean',
                'temp': 'mean',
                'weathersit': 'mean',
                'casual': 'mean',
                'registered': 'mean'
            }).round(2)
            
            fig_cluster_chars = px.imshow(
                cluster_stats.T,
                title="Cluster Characteristics Heatmap",
                labels={'x': 'Clusters', 'y': 'Features', 'color': 'Average Value'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_cluster_chars, use_container_width=True)
        
        # Cluster summary
        st.subheader("Cluster Analysis Summary")
        cluster_summary = day_df.groupby('usage_cluster').agg({
            'cnt': ['count', 'mean', 'std'],
            'temp': 'mean',
            'weathersit': 'mean'
        }).round(2)
        
        cluster_summary.columns = ['Days_Count', 'Avg_Usage', 'Usage_StdDev', 'Avg_Temp', 'Avg_Weather']
        st.dataframe(cluster_summary, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in Clustering Analysis: {str(e)}")
        st.info("Using simplified clustering approach.")
        
        # Simplified clustering as fallback
        day_df['usage_cluster'] = pd.cut(day_df['cnt'], bins=3, labels=['Low Usage', 'Medium Usage', 'High Usage'])
        
        simple_summary = day_df.groupby('usage_cluster').agg({
            'cnt': ['count', 'mean'],
            'temp': 'mean'
        }).round(2)
        simple_summary.columns = ['Days_Count', 'Avg_Usage', 'Avg_Temp']
        
        st.subheader("Simplified Usage Clustering")
        st.dataframe(simple_summary, use_container_width=True)
    
    return day_df

def create_advanced_visualizations(day_df, hour_df):
    """Create advanced visualizations with better design principles"""
    
    st.markdown('<h2 class="section-header">📈 Advanced Data Visualizations</h2>', unsafe_allow_html=True)
    
    # 1. Time Series Decomposition View
    st.subheader("📅 Time Series Analysis")
    
    # Monthly trend with seasonal decomposition view
    monthly_data = day_df.groupby(['year', 'month']).agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
    
    fig_ts = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Total Monthly Usage', 'User Type Breakdown'),
        vertical_spacing=0.1
    )
    
    # Total usage trend
    fig_ts.add_trace(
        go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['cnt'],
            mode='lines+markers',
            name='Total Usage',
            line=dict(width=3, color='#1f77b4')
        ),
        row=1, col=1
    )
    
    # Stacked area for user types
    fig_ts.add_trace(
        go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['casual'],
            fill='tonexty',
            mode='none',
            name='Casual Users',
            fillcolor='rgba(255, 127, 14, 0.6)'
        ),
        row=2, col=1
    )
    
    fig_ts.add_trace(
        go.Scatter(
            x=monthly_data['date'],
            y=monthly_data['casual'] + monthly_data['registered'],
            fill='tonexty',
            mode='none',
            name='Registered Users',
            fillcolor='rgba(31, 119, 180, 0.6)'
        ),
        row=2, col=1
    )
    
    fig_ts.update_layout(height=600, title_text="Time Series Analysis Dashboard")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # 2. Multi-dimensional Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Weather vs Usage bubble chart
        weather_analysis = day_df.groupby(['weathersit_name', 'temp_category']).agg({
            'cnt': ['mean', 'count'],
            'casual': 'mean',
            'registered': 'mean'
        }).reset_index()
        
        weather_analysis.columns = ['weather', 'temp_cat', 'avg_usage', 'count_days', 'avg_casual', 'avg_registered']
        
        fig_bubble = px.scatter(
            weather_analysis,
            x='avg_casual',
            y='avg_registered',
            size='count_days',
            color='weather',
            hover_data=['temp_cat', 'avg_usage'],
            title="Weather Impact on User Types",
            labels={
                'avg_casual': 'Average Casual Users',
                'avg_registered': 'Average Registered Users'
            }
        )
        
        fig_bubble.update_traces(marker=dict(opacity=0.7, sizemode='diameter', sizeref=2.*max(weather_analysis['count_days'])/(80**2)))
        st.plotly_chart(fig_bubble, use_container_width=True)
    
    with col2:
        # Seasonal performance radar chart
        seasonal_metrics = day_df.groupby('season_name').agg({
            'cnt': 'mean',
            'casual': 'mean',
            'registered': 'mean',
            'temp': 'mean',
            'hum': 'mean'
        }).reset_index()
        
        # Normalize metrics for radar chart
        for col in ['cnt', 'casual', 'registered', 'temp', 'hum']:
            seasonal_metrics[f'{col}_norm'] = (seasonal_metrics[col] - seasonal_metrics[col].min()) / (seasonal_metrics[col].max() - seasonal_metrics[col].min())
        
        fig_radar = go.Figure()
        
        seasons = seasonal_metrics['season_name'].tolist()
        metrics = ['cnt_norm', 'casual_norm', 'registered_norm', 'temp_norm', 'hum_norm']
        metric_names = ['Total Usage', 'Casual Users', 'Registered Users', 'Temperature', 'Humidity']
        
        for season in seasons:
            season_data = seasonal_metrics[seasonal_metrics['season_name'] == season]
            values = [season_data[metric].iloc[0] for metric in metrics]
            values += [values[0]]  # Close the polygon
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_names + [metric_names[0]],
                fill='toself',
                name=season
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Seasonal Performance Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

def main():
    # Load data
    day_df, hour_df = load_data()
    
 # Sidebar - Date Range Filter
    st.sidebar.header("📅 Filter Tanggal")

    # Ambil batas tanggal dari dataset
    min_date = day_df['dteday'].min()  # 2011-01-01
    max_date = day_df['dteday'].max()  # 2012-12-31

    # Date picker (dibatasi otomatis oleh min/max dataset)
    start_date, end_date = st.sidebar.date_input(
        "Pilih rentang tanggal:",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    # Validasi input tanggal
    if start_date > end_date:
        st.error("❌ Tanggal mulai tidak boleh setelah tanggal akhir.")
        st.stop()

    filtered_day_df = day_df[(day_df['dteday'] >= pd.to_datetime(start_date)) & (day_df['dteday'] <= pd.to_datetime(end_date))]
    filtered_hour_df = hour_df[(hour_df['dteday'] >= pd.to_datetime(start_date)) & (hour_df['dteday'] <= pd.to_datetime(end_date))]

    # Enhanced Header with animation effect
    st.markdown('<h1 class="main-header">🚴‍♂️ Advanced Bike Sharing Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.3em; color: #666; font-style: italic;">'
        'Comprehensive Analysis of Bike Sharing System (2011-2012) with Advanced Analytics</p>', 
        unsafe_allow_html=True
    )
    
    # Enhanced metrics with better design
    st.markdown('<h2 class="section-header">📊 Key Performance Indicators</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_rentals = day_df['cnt'].sum()
        st.metric("🚴 Total Rentals", f"{total_rentals:,}", delta="100% of data")
    
    with col2:
        avg_daily = day_df['cnt'].mean()
        growth_rate = ((day_df[day_df['year'] == day_df['year'].max()]['cnt'].mean() - 
                       day_df[day_df['year'] == day_df['year'].min()]['cnt'].mean()) / 
                      day_df[day_df['year'] == day_df['year'].min()]['cnt'].mean() * 100)
        st.metric("📈 Avg Daily Usage", f"{avg_daily:.0f}", delta=f"{growth_rate:.1f}% YoY")
    
    with col3:
        peak_day = day_df['cnt'].max()
        peak_date = day_df.loc[day_df['cnt'].idxmax(), 'dteday'].strftime('%Y-%m-%d')
        st.metric("🎯 Peak Day Usage", f"{peak_day:,}", delta=f"on {peak_date}")
    
    with col4:
        casual_pct = (day_df['casual'].sum() / total_rentals * 100)
        st.metric("👥 Casual Users %", f"{casual_pct:.1f}%", delta="vs Registered")
    
    with col5:
        weather_impact = day_df.groupby('weathersit')['cnt'].mean()
        weather_efficiency = (weather_impact[1] - weather_impact[3]) / weather_impact[1] * 100
        st.metric("🌤️ Weather Impact", f"{weather_efficiency:.0f}%", delta="Clear vs Rainy")
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Core Analytics", "🔍 Advanced Analytics", "🎯 RFM Analysis", "📈 Clustering Analysis"])
    
    with tab1:
        # Core analysis (your existing code)
        st.markdown('<h2 class="section-header">🌟 Seasonal and Weather Impact Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            seasonal_data = day_df.groupby('season_name')['cnt'].mean().sort_values(ascending=False).reset_index()
            
            fig_season = px.bar(
                seasonal_data,
                x='season_name',
                y='cnt',
                title="Average Rentals by Season",
                labels={'season_name': 'Season', 'cnt': 'Average Daily Rentals'},
                color='cnt',
                color_continuous_scale='viridis',
                text='cnt'
            )
            fig_season.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig_season.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_season, use_container_width=True)
        
        with col2:
            weather_data = day_df.groupby('weathersit_name')['cnt'].mean().sort_values(ascending=False).reset_index()
            
            fig_weather = px.bar(
                weather_data,
                x='weathersit_name',
                y='cnt',
                title="Average Rentals by Weather Condition",
                labels={'weathersit_name': 'Weather Condition', 'cnt': 'Average Daily Rentals'},
                color='cnt',
                color_continuous_scale='plasma',
                text='cnt'
            )
            fig_weather.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig_weather.update_layout(showlegend=False, height=400)
            fig_weather.update_xaxes(tickangle=45)
            st.plotly_chart(fig_weather, use_container_width=True)
        
        # Temporal patterns
        st.markdown('<h2 class="section-header">⏰ Temporal Usage Patterns</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weekly pattern
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_avg = day_df.groupby('weekday_name')['cnt'].mean().reindex(weekday_order)
            
            fig_weekly = px.line(
                x=daily_avg.index,
                y=daily_avg.values,
                title="Weekly Usage Pattern",
                labels={'x': 'Day of Week', 'y': 'Average Rentals'},
                markers=True
            )
            fig_weekly.update_traces(line=dict(width=4), marker=dict(size=8))
            fig_weekly.update_layout(height=400)
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        with col2:
            # Weekday vs Weekend comparison
            weekend_comparison = day_df.copy()
            weekend_comparison['day_type'] = weekend_comparison['day_of_week'].apply(
                lambda x: 'Weekend' if x >= 5 else 'Weekday'
            )
            comparison_data = weekend_comparison.groupby('day_type').agg({
                'casual': 'mean',
                'registered': 'mean'
            }).reset_index()
            
            fig_comparison = px.bar(
                comparison_data,
                x='day_type',
                y=['casual', 'registered'],
                title="Weekday vs Weekend Usage",
                labels={'value': 'Average Rentals', 'variable': 'User Type'},
                barmode='group'
            )
            fig_comparison.update_layout(height=400)
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Hourly patterns
        if not hour_df.empty:
            st.subheader("🕐 Hourly Usage Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                hourly_avg = hour_df.groupby('hr')['cnt'].mean()
                
                fig_hourly = px.line(
                    x=hourly_avg.index,
                    y=hourly_avg.values,
                    title="Average Hourly Usage Pattern",
                    labels={'x': 'Hour of Day', 'y': 'Average Rentals'},
                    markers=True
                )
                fig_hourly.update_traces(line=dict(width=3), marker=dict(size=6))
                fig_hourly.update_layout(height=400)
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col2:
                # Weekday vs Weekend hourly
                hourly_comparison = hour_df.groupby(['hr', 'is_weekend'])['cnt'].mean().unstack()
                
                fig_hourly_comp = go.Figure()
                if 'Weekday' in hourly_comparison.columns:
                    fig_hourly_comp.add_trace(go.Scatter(
                        x=hourly_comparison.index,
                        y=hourly_comparison['Weekday'],
                        name='Weekday',
                        line=dict(width=3, color='#1f77b4')
                    ))
                if 'Weekend' in hourly_comparison.columns:
                    fig_hourly_comp.add_trace(go.Scatter(
                        x=hourly_comparison.index,
                        y=hourly_comparison['Weekend'],
                        name='Weekend',
                        line=dict(width=3, color='#ff7f0e')
                    ))
                
                fig_hourly_comp.update_layout(
                    title="Hourly Pattern: Weekday vs Weekend",
                    xaxis_title="Hour of Day",
                    yaxis_title="Average Rentals",
                    height=400
                )
                st.plotly_chart(fig_hourly_comp, use_container_width=True)
    
    with tab2:
        create_advanced_visualizations(day_df, hour_df)
        
        # Environmental correlation analysis
        st.subheader("🌡️ Environmental Factors Correlation")
        correlation_vars = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
        if all(col in day_df.columns for col in correlation_vars):
            corr_matrix = day_df[correlation_vars].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Environmental Factors Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto',
                text_auto=True
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        rfm_results = perform_rfm_analysis(day_df)
        
        # RFM Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🎯 RFM Analysis Insights:**")
        st.markdown("""
        - **Champions**: Periods with consistent high usage and recent activity
        - **Loyal Customers**: Reliable usage patterns with good frequency
        - **At Risk**: Previously good periods but declining recent activity
        - **Potential Loyalists**: Good recent activity, can be developed further
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        clustering_results = perform_clustering_analysis(day_df)
        
        # Clustering Insights
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**📊 Clustering Analysis Insights:**")
        st.markdown("""
        - **Peak Performance**: Optimal conditions with highest usage
        - **Weather Affected**: Low usage due to poor weather conditions
        - **Weekend Warriors**: High weekend recreational usage
        - **Regular Commuters**: Consistent weekday commuting patterns
        - **Hot Weather Users**: Moderate usage during hot conditions
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Key Insights Section
    st.markdown('<h2 class="section-header">💡 Comprehensive Business Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🌤️ Environmental Impact Analysis:**")
        
        # Calculate specific insights
        season_performance = day_df.groupby('season_name')['cnt'].mean()
        best_season = season_performance.idxmax()
        worst_season = season_performance.idxmin()
        season_difference = ((season_performance[best_season] - season_performance[worst_season]) / 
                           season_performance[worst_season] * 100)
        
        weather_performance = day_df.groupby('weathersit')['cnt'].mean()
        weather_impact = ((weather_performance[1] - weather_performance[3]) / weather_performance[1] * 100)
        
        st.markdown(f"""
        - **{best_season}** is the peak season with {season_difference:.0f}% higher usage than **{worst_season}**
        - Clear weather generates **{weather_impact:.0f}% more** rentals than rainy conditions
        - Temperature correlation with usage: **{day_df['temp'].corr(day_df['cnt']):.2f}**
        - Optimal temperature range: **{day_df[day_df['cnt'] >= day_df['cnt'].quantile(0.8)]['temp'].mean():.2f}** (normalized)
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**⏰ Temporal Usage Insights:**")
        
        # Calculate temporal insights
        if not hour_df.empty:
            peak_hour = hour_df.groupby('hr')['cnt'].mean().idxmax()
            low_hour = hour_df.groupby('hr')['cnt'].mean().idxmin()
            
            weekday_avg = day_df[day_df['weekday'] < 5]['cnt'].mean()
            weekend_avg = day_df[day_df['weekday'] >= 5]['cnt'].mean()
            weekend_diff = ((weekend_avg - weekday_avg) / weekday_avg * 100)
        else:
            peak_hour = 17
            low_hour = 3
            weekend_diff = -15
        
        registered_dominance = (day_df['registered'].sum() / day_df['cnt'].sum() * 100)
        
        st.markdown(f"""
        - Peak usage occurs at **{peak_hour}:00** (rush hour pattern)
        - Lowest activity at **{low_hour}:00** (optimal for maintenance)
        - Weekend usage is **{abs(weekend_diff):.0f}%** {'higher' if weekend_diff > 0 else 'lower'} than weekdays
        - Registered users dominate with **{registered_dominance:.0f}%** of total usage
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategic Recommendations
    st.markdown('<h2 class="section-header">🎯 Strategic Business Recommendations</h2>', unsafe_allow_html=True)
    
    recommendations = f"""
    **🚀 Operational Optimization:**
    
    **📈 Capacity Management:**
    - Increase bike availability during **{best_season}** season (+{season_difference:.0f}% demand)
    - Deploy 40% more bikes during peak hours (17:00) and rush periods
    - Reduce fleet size during low-demand periods (3:00-6:00) for cost efficiency
    
    **🌦️ Weather-Based Strategy:**
    - Implement dynamic pricing: premium rates during clear weather
    - Develop covered stations for light rain/snow conditions
    - Create weather-based promotional campaigns
    
    **👥 User Segmentation:**
    - **Registered Users ({registered_dominance:.0f}% of usage)**: Focus on loyalty programs and commuter packages
    - **Casual Users**: Implement weekend recreational packages and tourist-friendly features
    - Develop mobile app features for different user behaviors
    
    **📊 Advanced Analytics Applications:**
    - **RFM Segmentation**: Identify high-value time periods for targeted interventions
    - **Usage Clustering**: Optimize bike distribution based on pattern recognition
    - **Predictive Maintenance**: Schedule during identified low-usage periods
    
    **💰 Revenue Optimization:**
    - Implement surge pricing during peak demand periods
    - Develop seasonal membership plans aligned with usage patterns
    - Create corporate partnerships for registered user acquisition
    """
    
    st.markdown(recommendations)
    
    # Technical Implementation Notes
    st.markdown('<h2 class="section-header">🔧 Technical Implementation & Methodology</h2>', unsafe_allow_html=True)
    
    methodology = """
    **📋 Advanced Analytics Methodology:**
    
    **🔍 RFM Analysis Adaptation:**
    - **Recency**: Days since last high-usage period (adapted for time-series data)
    - **Frequency**: Number of high-usage days per month
    - **Monetary**: Total usage volume (proxy for revenue generation)
    - **Segmentation**: 6 distinct behavioral segments identified
    - **Robust Binning**: Error handling for insufficient data diversity
    
    **📊 Manual Clustering Approach:**
    - **Business Rule-Based**: No ML algorithms, pure domain knowledge
    - **Multi-dimensional**: Usage, weather, temperature, day type consideration
    - **6 Usage Patterns**: Peak Performance, Weather Affected, Weekend Warriors, etc.
    - **Validation**: Cross-referenced with seasonal and temporal patterns
    
    **📈 Visualization Design Principles:**
    - **Accessibility**: High contrast colors, clear labeling
    - **Interactivity**: Hover details, responsive design
    - **Data Integrity**: Accurate scaling, proper aggregation
    - **Progressive Disclosure**: Information hierarchy through tabs
    
    **🚀 Deployment Considerations:**
    - Streamlit Cloud compatible
    - Responsive design for mobile devices
    - Cached data loading for performance
    - Error handling for missing datasets
    - Robust statistical functions with fallback options
    """
    
    with st.expander("View Technical Details"):
        st.markdown(methodology)
    
    # Data Quality & Sources
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-style: italic;">'
        '📊 Dashboard built with advanced analytics • RFM Analysis • Manual Clustering • Enhanced Visualizations<br>'
        'Data Period: 2011-2012 • Built with Streamlit & Plotly • Designed for Business Intelligence</p>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()