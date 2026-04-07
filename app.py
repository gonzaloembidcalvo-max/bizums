import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Bizum Analytics", page_icon="💸", layout="wide")

FRIENDS = ['Tom', 'Edu', 'Victor', 'Cesar', 'Quique', 'Pablo', 'Jorge', 'Tomi', 'Gonza', 'Alonso', 'Dias', 'Alex']
DATA_FOLDER = 'data'

@st.cache_data
def load_data():
    try:
        all_dfs = []
        for filename in os.listdir(DATA_FOLDER):
            if filename.endswith('.csv'):
                filepath = os.path.join(DATA_FOLDER, filename)
                df_temp = pd.read_csv(filepath)
                all_dfs.append(df_temp)
        
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            df = df.dropna(subset=['Fecha'])
            df = df.sort_values('Fecha').reset_index(drop=True)
            return df
        return None
    except Exception as e:
        return None

@st.cache_data
def analyze_data(df):
    if df is None:
        return {}
    
    stats = {}
    stats['total_bizums'] = len(df)
    stats['total_amount'] = df['Cantidad'].sum()
    stats['avg_per_day'] = len(df) / df['Fecha'].nunique()
    stats['avg_per_week'] = len(df) / max((df['Fecha'].max() - df['Fecha'].min()).days, 1) * 7
    stats['avg_per_month'] = len(df) / max((df['Fecha'].max() - df['Fecha'].min()).days, 1) * 30
    
    stats['top_sender'] = df['Emisor'].value_counts().idxmax()
    stats['top_sender_count'] = df['Emisor'].value_counts().max()
    stats['most_received'] = df['Receptor'].value_counts().idxmax()
    
    yearly = df.groupby(df['Fecha'].dt.year).size()
    if len(yearly) > 1:
        years = yearly.index.tolist()
        stats['year_growth'] = ((yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2]) * 100
        stats['current_year'] = years[-1]
        stats['prev_year'] = years[-2]
        stats['current_count'] = yearly.iloc[-1]
        stats['prev_count'] = yearly.iloc[-2]
    
    stats['friend_stats'] = {}
    for friend in FRIENDS:
        sent = len(df[df['Emisor'] == friend])
        received = len(df[df['Receptor'] == friend])
        amount_sent = df[df['Emisor'] == friend]['Cantidad'].sum()
        stats['friend_stats'][friend] = {
            'sent': sent,
            'received': received,
            'total': sent + received,
            'amount_sent': amount_sent,
        }
    
    df['DiaSemana'] = df['Fecha'].dt.day_name()
    stats['weekday_probs'] = df.groupby('DiaSemana').size().to_dict()
    
    return stats

@st.cache_data
def get_forecast(df, friend):
    friend_df = df[df['Emisor'] == friend].copy()
    if len(friend_df) < 5:
        return None
    
    friend_df = friend_df.groupby('Fecha').size().reset_index(name='count')
    friend_df['DiaNum'] = (friend_df['Fecha'] - friend_df['Fecha'].min()).dt.days
    
    X = friend_df['DiaNum'].values.reshape(-1, 1)
    y = friend_df['count'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    first_date = df['Fecha'].min()
    last_date = df['Fecha'].max()
    days_elapsed = (last_date - first_date).days
    
    end_of_year = pd.Timestamp(f'{first_date.year}-12-31')
    days_total = (end_of_year - first_date).days
    
    X_pred = np.array([[days_total]]).reshape(-1, 1)
    predicted_total = model.predict(X_pred)[0]
    
    days_to_100 = None
    if model.coef_[0] > 0 and friend_df['count'].sum() < 100:
        current_total = friend_df['count'].sum()
        remaining = 100 - current_total
        days_to_100 = remaining / (model.coef_[0] * (len(friend_df) / max(days_elapsed, 1))) if days_elapsed > 0 else None
    
    return {
        'current': int(friend_df['count'].sum()),
        'predicted': int(max(predicted_total, friend_df['count'].sum())),
        'daily_rate': model.coef_[0] if model.coef_[0] > 0 else 0,
        'days_to_100': days_to_100,
    }

def get_conditional_prob(df, friend, weekday):
    weekday_df = df[df['Fecha'].dt.day_name() == weekday]
    total_weekday = len(weekday_df)
    if total_weekday == 0:
        return 0
    friend_weekday = len(weekday_df[weekday_df['Emisor'] == friend])
    return (friend_weekday / total_weekday) * 100

def create_cumulative_chart(df):
    df_grouped = df.groupby('Fecha').size().cumsum().reset_index()
    df_grouped.columns = ['Fecha', 'Acumulado']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_grouped['Fecha'],
        y=df_grouped['Acumulado'],
        mode='lines',
        fill='tozeroy',
        line=dict(color='#007AFF', width=3),
        fillcolor='rgba(0,122,255,0.2)',
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Bizums: %{y}<extra></extra>',
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, -apple-system, sans-serif', color='#1D1D1F'),
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(showgrid=False, zeroline=False, color='#86868B'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', zeroline=False, color='#86868B'),
        hovermode='x unified',
    )
    return fig

def create_heatmap(df):
    df = df.copy()
    df['DiaSemana'] = df['Fecha'].dt.day_name()
    df['Mes'] = df['Fecha'].dt.month_name()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    pivot = df.groupby(['DiaSemana', 'Mes']).size().unstack(fill_value=0)
    pivot = pivot.reindex(day_order)
    pivot = pivot.reindex(columns=[c for c in month_order if c in pivot.columns])
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'][:len(pivot.index)],
        colorscale=[[0, '#F5F5F7'], [0.5, 'rgba(0,122,255,0.6)'], [1, '#007AFF']],
        showscale=False,
        hovertemplate='<b>%{y}</b> %{x}<br>Bizums: %{z}<extra></extra>',
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#1D1D1F'),
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(showgrid=False, color='#86868B', tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, color='#86868B', tickfont=dict(size=10)),
    )
    return fig

def create_weekly_chart(df):
    df = df.copy()
    df['DiaSemana'] = df['Fecha'].dt.day_name()
    week_counts = df.groupby('DiaSemana').size()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_labels = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    counts = [week_counts.get(d, 0) for d in day_order]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=day_labels,
        y=counts,
        marker_color=['#007AFF' if c == max(counts) else 'rgba(88,86,214,0.6)' for c in counts],
        hovertemplate='<b>%{x}</b><br>Bizums: %{y}<extra></extra>',
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#1D1D1F'),
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
        xaxis=dict(showgrid=False, color='#86868B'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', color='#86868B'),
    )
    return fig

def create_forecast_chart(df):
    fig = go.Figure()
    
    for friend in df['Emisor'].unique():
        friend_df = df[df['Emisor'] == friend].groupby('Fecha').size().cumsum().reset_index()
        if len(friend_df) > 5:
            fig.add_trace(go.Scatter(
                x=friend_df['Fecha'],
                y=friend_df.iloc[:, 1],
                mode='lines',
                name=friend,
                line=dict(width=2),
                hovertemplate=f'{friend}: %{{y}} bizums<br>%{{x|%d %b}}<extra></extra>',
            ))
    
    last_date = df['Fecha'].max()
    current_day = (last_date - df['Fecha'].min()).days
    
    if current_day > 0 and len(df) > 10:
        total_bizums = len(df)
        daily_rate = total_bizums / current_day
        
        projected_dates = pd.date_range(start=last_date, periods=30, freq='D')
        projected_values = [total_bizums + daily_rate * i for i in range(1, 31)]
        
        fig.add_trace(go.Scatter(
            x=projected_dates,
            y=projected_values,
            mode='lines',
            name='Proyección',
            line=dict(color='#FF9500', width=2, dash='dash'),
            hovertemplate='Proyectado: %{y:.0f} bizums<br>%{x|%d %b}<extra></extra>',
        ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#1D1D1F'),
        height=380,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        xaxis=dict(showgrid=False, color='#86868B'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', color='#86868B'),
        hovermode='x unified',
    )
    return fig

def create_affinity_matrix(df):
    friends_subset = [f for f in FRIENDS if f in df['Emisor'].values or f in df['Receptor'].values]
    
    if len(friends_subset) < 2:
        return None
    
    affinity = pd.DataFrame(0, index=friends_subset, columns=friends_subset)
    
    for _, row in df.iterrows():
        if row['Emisor'] in friends_subset and row['Receptor'] in friends_subset:
            affinity.loc[row['Emisor'], row['Receptor']] += 1
    
    fig = go.Figure(data=go.Heatmap(
        z=affinity.values,
        x=affinity.columns,
        y=affinity.index,
        colorscale='Blues',
        showscale=False,
        hovertemplate='<b>%{y}</b> → %{x}<br>Interacciones: %{z}<extra></extra>',
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#1D1D1F', size=10),
        height=400,
        margin=dict(l=60, r=20, t=20, b=40),
        xaxis=dict(tickangle=45, showgrid=False, color='#86868B'),
        yaxis=dict(showgrid=False, color='#86868B'),
    )
    return fig

def create_ranking_chart(friend_stats, sort_by='sent'):
    sorted_friends = sorted(friend_stats.items(), key=lambda x: x[1][sort_by], reverse=True)[:8]
    
    names = [f[0] for f in sorted_friends]
    values = [f[1][sort_by] for f in sorted_friends]
    
    colors = ['#007AFF' if i < 3 else 'rgba(88,86,214,0.6)' for i in range(len(names))]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation='h',
        marker_color=colors,
        hovertemplate='<b>%{y}</b><br>%{x} bizums<extra></extra>',
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#1D1D1F'),
        height=350,
        margin=dict(l=80, r=20, t=20, b=40),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)', color='#86868B'),
        yaxis=dict(showgrid=False, color='#86868B'),
    )
    return fig

css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #F5F5F7 0%, #FFFFFF 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        font-size: 32px;
        font-weight: 700;
        color: #1D1D1F;
        margin-bottom: 8px;
    }
    
    .sub-header {
        font-size: 14px;
        color: #86868B;
        margin-bottom: 24px;
    }
    
    div[data-testid="stHorizontalBlock"] > div {
        background: rgba(255,255,255,0.72);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    }
    
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    
    .stMetric {
        background: white;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(0,0,0,0.06);
    }
    
    .stMetric label {
        color: #86868B !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #1D1D1F !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1D1D1F;
        margin-bottom: 16px;
        margin-top: 24px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.5);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #007AFF !important;
        color: white !important;
    }
    
    .insight-card {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        border-radius: 16px;
        padding: 16px;
        color: white;
    }
    
    .stAlert {
        border-radius: 12px;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(20px);
    }
    
    .nav-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 10px 20px;
        border-radius: 12px;
        background: #007AFF;
        color: white;
        font-weight: 600;
        margin: 4px;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .nav-button:hover {
        background: #0056b3;
    }
    
    .nav-button.active {
        background: #007AFF;
        box-shadow: 0 4px 12px rgba(0,122,255,0.3);
    }
</style>
"""

st.markdown(css, unsafe_allow_html=True)

df = load_data()
stats = analyze_data(df) if df is not None else {}

st.markdown('<p class="main-header">💸 Bizum Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Análisis de actividad Bizum entre amigos</p>', unsafe_allow_html=True)

if df is None:
    st.warning("⚠️ No se encontraron archivos CSV en la carpeta `data/`. Añade tus archivos y reinicia.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🏆 Rankings", "📈 Tendencias", "🔮 Proyecciones", "📉 Probabilidades"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bizums", f"{stats['total_bizums']}", f"{stats['avg_per_day']:.1f} / día")
    with col2:
        st.metric("Cantidad Total", f"€{stats['total_amount']:.0f}", f"€{stats['total_amount']/max(stats['total_bizums'],1):.1f} avg")
    with col3:
        st.metric("Promedio Semanal", f"{stats['avg_per_week']:.1f}", "bizums")
    with col4:
        st.metric("Promedio Mensual", f"{stats['avg_per_month']:.1f}", "bizums")
    
    st.markdown('<p class="section-title">📊 Top Enviadores</p>', unsafe_allow_html=True)
    
    top_senders = sorted(stats['friend_stats'].items(), key=lambda x: x[1]['sent'], reverse=True)[:5]
    for i, (friend, data) in enumerate(top_senders, 1):
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.markdown(f"**#{i}**")
        with col2:
            progress = data['sent'] / max(max([s[1]['sent'] for s in top_senders]), 1)
            st.progress(progress, text=friend)
        with col3:
            st.caption(f"{data['sent']} bizums")
    
    st.markdown('<p class="section-title">📈 Evolución Acumulada</p>', unsafe_allow_html=True)
    st.plotly_chart(create_cumulative_chart(df), use_container_width=True)
    
    if 'year_growth' in stats:
        col1, col2 = st.columns(2)
        with col1:
            trend = "📈" if stats['year_growth'] > 0 else "📉"
            st.info(f"{trend} {stats['prev_year']} vs {stats['current_year']}: {stats['year_growth']:+.1f}%")
        with col2:
            st.info(f"📊 {stats['prev_count']} → {stats['current_count']} bizums")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="section-title">🏆 Top Enviadores</p>', unsafe_allow_html=True)
        st.plotly_chart(create_ranking_chart(stats['friend_stats'], 'sent'), use_container_width=True)
    
    with col2:
        st.markdown('<p class="section-title">🎁 Top Recibidores</p>', unsafe_allow_html=True)
        st.plotly_chart(create_ranking_chart(stats['friend_stats'], 'received'), use_container_width=True)
    
    st.markdown('<p class="section-title">📋 Tabla Completa</p>', unsafe_allow_html=True)
    
    table_data = []
    sorted_total = sorted(stats['friend_stats'].items(), key=lambda x: x[1]['total'], reverse=True)
    for i, (friend, data) in enumerate(sorted_total, 1):
        table_data.append({
            "#": f"#{i}",
            "Amigo": friend,
            "Enviados": data['sent'],
            "Recibidos": data['received'],
            "Total €": f"€{data['amount_sent']:.0f}"
        })
    
    st.dataframe(table_data, use_container_width=True, hide_index=True)

with tab3:
    st.markdown('<p class="section-title">🗓️ Mapa de Calor: Día de la semana</p>', unsafe_allow_html=True)
    st.plotly_chart(create_heatmap(df), use_container_width=True)
    
    st.markdown('<p class="section-title">📊 Actividad por Día</p>', unsafe_allow_html=True)
    st.plotly_chart(create_weekly_chart(df), use_container_width=True)
    
    st.markdown('<p class="section-title">⚠️ Detección de Picos</p>', unsafe_allow_html=True)
    
    df_copy = df.copy()
    df_copy['FechaStr'] = df_copy['Fecha'].dt.date
    daily_counts = df_copy.groupby('FechaStr').size()
    mean = daily_counts.mean()
    std = daily_counts.std()
    threshold = mean + 1.5 * std
    anomalies = daily_counts[daily_counts > threshold]
    
    if len(anomalies) > 0:
        for date, count in anomalies.items():
            st.warning(f"🚨 Pico detectado el **{date}**: **{count}** bizums")
    else:
        st.success("✅ No se detectaron picos significativos")

with tab4:
    st.markdown('<p class="section-title">📈 Evolución y Proyección</p>', unsafe_allow_html=True)
    st.plotly_chart(create_forecast_chart(df), use_container_width=True)
    
    st.markdown('<p class="section-title">🔮 Predicciones por Persona (Fin de Año)</p>', unsafe_allow_html=True)
    
    forecast_data = []
    for friend in FRIENDS:
        if friend in df['Emisor'].values:
            fc = get_forecast(df, friend)
            if fc:
                forecast_data.append((friend, fc))
    
    forecast_data.sort(key=lambda x: x[1]['predicted'], reverse=True)
    
    for friend, fc in forecast_data[:6]:
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{friend}**")
                progress = min(fc['predicted'] / 100, 1.0)
                st.progress(progress, text=f"{fc['current']} → {fc['predicted']}")
            with col2:
                st.metric("Actual", fc['current'])
            with col3:
                st.metric("Proyectado", fc['predicted'])
    
    st.markdown('<p class="section-title">🏅 Hitos - ¿Quién alcanza 100 primero?</p>', unsafe_allow_html=True)
    
    milestones = []
    for friend, fc in forecast_data:
        if fc['days_to_100'] and 0 < fc['days_to_100'] < 365:
            milestones.append((friend, fc['days_to_100']))
    
    milestones.sort(key=lambda x: x[1])
    
    if milestones:
        for i, (friend, days) in enumerate(milestones[:3], 1):
            target_date = (datetime.now() + timedelta(days=int(days))).strftime('%d %b %Y')
            st.success(f"🥇 **{friend}**: ~{int(days)} días → {target_date}")
    else:
        st.info("No se prevén hitos de 100 bizums este año")

with tab5:
    st.markdown('<p class="section-title">📊 Probabilidad Condicional por Día</p>', unsafe_allow_html=True)
    
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_names = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    
    probs_data = []
    for friend in ['Tom', 'Edu', 'Victor', 'Cesar', 'Quique']:
        if friend in df['Emisor'].values:
            probs = [get_conditional_prob(df, friend, day) for day in weekdays]
            probs_data.append((friend, probs))
    
    prob_df = pd.DataFrame({
        name: probs for name, probs in probs_data
    }, index=weekday_names).T
    
    st.dataframe(prob_df.style.format("{:.1f}%").background_gradient(cmap='Blues'), use_container_width=True)
    
    st.markdown('<p class="section-title">🔗 Matriz de Afinidad</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#86868B;font-size:12px">¿Quién interactúa más con quién?</p>', unsafe_allow_html=True)
    
    affinity_fig = create_affinity_matrix(df)
    if affinity_fig:
        st.plotly_chart(affinity_fig, use_container_width=True)
    
    st.markdown('<p class="section-title">⭐ Análisis de Patrones</p>', unsafe_allow_html=True)
    
    if stats['weekday_probs']:
        max_day = max(stats['weekday_probs'], key=stats['weekday_probs'].get)
        day_translations = {
            'Monday': 'lunes', 'Tuesday': 'martes', 'Wednesday': 'miércoles',
            'Thursday': 'jueves', 'Friday': 'viernes', 'Saturday': 'sábado', 'Sunday': 'domingo'
        }
        st.info(f"🎯 El **{day_translations.get(max_day, max_day)}** es el día más activo con **{stats['weekday_probs'][max_day]}** bizums")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#86868B;font-size:12px'>Bizum Analytics • Diseño iOS 26 Style</p>", unsafe_allow_html=True)
