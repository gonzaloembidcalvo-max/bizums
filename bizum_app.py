import flet as ft
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import warnings
import os
warnings.filterwarnings('ignore')

FRIENDS = ['Tom', 'Edu', 'Victor', 'Cesar', 'Quique', 'Pablo', 'Jorge', 'Tomi', 'Gonza', 'Alonso', 'Dias', 'Alex']
DATA_FOLDER = 'data'

COLORS_LIGHT = {
    'bg': '#F5F5F7',
    'card': '#FFFFFF',
    'card_glass': 'rgba(255,255,255,0.72)',
    'text': '#1D1D1F',
    'text_secondary': '#86868B',
    'primary': '#007AFF',
    'secondary': '#5856D6',
    'success': '#34C759',
    'warning': '#FF9500',
    'destructive': '#FF3B30',
    'border': 'rgba(0,0,0,0.06)',
}

COLORS_DARK = {
    'bg': '#000000',
    'card': '#1C1C1E',
    'card_glass': 'rgba(28,28,30,0.72)',
    'text': '#F5F5F7',
    'text_secondary': '#98989D',
    'primary': '#0A84FF',
    'secondary': '#5E5CE6',
    'success': '#30D158',
    'warning': '#FF9F0A',
    'destructive': '#FF453A',
    'border': 'rgba(255,255,255,0.06)',
}

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
        print(f"Error loading data: {e}")
        return None

def glass_card(content, width=None, height=None, padding=20):
    return ft.Container(
        content=content,
        width=width,
        height=height,
        padding=padding,
        border_radius=20,
        bgcolor=page.session.get('glass_color') or 'rgba(255,255,255,0.72)',
        border=ft.border.all(1, page.session.get('border_color') or 'rgba(0,0,0,0.06)'),
        blur=ft.Blur(20, 20),
    )

def get_colors():
    if page.theme_mode == ft.ThemeMode.DARK:
        return COLORS_DARK
    return COLORS_LIGHT

def stat_card(title, value, subtitle, icon_name, color_key):
    colors = get_colors()
    return glass_card(
        ft.Column([
            ft.Row([
                ft.Container(
                    ft.Icon(icon_name, size=24, color=colors[color_key]),
                    padding=10,
                    border_radius=10,
                    bgcolor=colors[color_key] + '15',
                ),
                ft.Column([
                    ft.Text(title, size=12, color=colors['text_secondary'], weight=ft.FontWeight.W_500),
                    ft.Text(value, size=28, weight=ft.FontWeight.BOLD, color=colors['text']),
                ], spacing=0, expand=True),
            ]),
            ft.Text(subtitle, size=12, color=colors['text_secondary']),
        ], spacing=8),
        padding=16,
    )

def ranking_row(name, rank, count, total_amount, max_count, is_top=True):
    colors = get_colors()
    bar_color = colors['success'] if is_top else colors['warning']
    
    return ft.Container(
        padding=12,
        border_radius=12,
        bgcolor=colors['card'],
        content=ft.Row([
            ft.Container(
                ft.Text(f'#{rank}', size=18, weight=ft.FontWeight.BOLD, color=bar_color),
                width=50,
            ),
            ft.Column([
                ft.Text(name, size=16, weight=ft.FontWeight.W_600, color=colors['text']),
                ft.Container(
                    height=6,
                    border_radius=3,
                    bgcolor=colors['bg'],
                    content=ft.Container(
                        width=min(count / max_count * 200, 200),
                        border_radius=3,
                        gradient=ft.LinearGradient(colors=[bar_color, bar_color + '80']),
                    ),
                ),
            ], spacing=4, expand=True),
            ft.Column([
                ft.Text(f'{count}', size=16, weight=ft.FontWeight.BOLD, color=colors['text']),
                ft.Text(f'{total_amount:.0f}€', size=12, color=colors['text_secondary']),
            ], horizontal_alignment=ft.CrossAxisAlignment.END),
        ], spacing=16, alignment=ft.MainAxisAlignment.START),
    )

def get_plotly_html(fig, height=300):
    return fig.to_html(full_html=False, include_plotlyjs='cdn', div_id='plot')

def create_cumulative_chart(df, colors):
    df_grouped = df.groupby('Fecha').size().cumsum().reset_index()
    df_grouped.columns = ['Fecha', 'Acumulado']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_grouped['Fecha'],
        y=df_grouped['Acumulado'],
        mode='lines',
        fill='tozeroy',
        line=dict(color=colors['primary'], width=3),
        fillcolor=colors['primary'] + '20',
        hovertemplate='<b>%{x|%d %b %Y}</b><br>Bizums: %{y}<extra></extra>',
    ))
    
    fig.update_layout(
        paper_bgcolor='transparent',
        plot_bgcolor='transparent',
        font=dict(family='Inter, sans-serif', color=colors['text']),
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(showgrid=False, zeroline=False, color=colors['text_secondary']),
        yaxis=dict(showgrid=True, gridcolor=colors['border'], zeroline=False, color=colors['text_secondary']),
        hovermode='x unified',
    )
    return get_plotly_html(fig, height)

def create_heatmap(df, colors):
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
        colorscale=[[0, colors['bg']], [0.5, colors['primary'] + '60'], [1, colors['primary']]],
        showscale=False,
        hovertemplate='<b>%{y}</b> %{x}<br>Bizums: %{z}<extra></extra>',
    ))
    
    fig.update_layout(
        paper_bgcolor='transparent',
        plot_bgcolor='transparent',
        font=dict(family='Inter, sans-serif', color=colors['text']),
        height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(showgrid=False, color=colors['text_secondary'], tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, color=colors['text_secondary'], tickfont=dict(size=10)),
    )
    return get_plotly_html(fig, 280)

def create_forecast_chart(df, colors):
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
    days_in_year = 365
    current_day = (last_date - df['Fecha'].min()).days
    
    if current_day > 0 and len(df) > 10:
        total_bizums = len(df)
        daily_rate = total_bizums / current_day
        remaining_days = days_in_year - current_day
        
        projected_total = total_bizums + (daily_rate * remaining_days)
        projected_dates = pd.date_range(start=last_date, periods=30, freq='D')
        projected_values = [total_bizums + daily_rate * i for i in range(1, 31)]
        
        fig.add_trace(go.Scatter(
            x=projected_dates,
            y=projected_values,
            mode='lines',
            name='Proyección',
            line=dict(color=colors['warning'], width=2, dash='dash'),
            hovertemplate='Proyectado: %{y:.0f} bizums<br>%{x|%d %b}<extra></extra>',
        ))
    
    fig.update_layout(
        paper_bgcolor='transparent',
        plot_bgcolor='transparent',
        font=dict(family='Inter, sans-serif', color=colors['text']),
        height=350,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        xaxis=dict(showgrid=False, color=colors['text_secondary']),
        yaxis=dict(showgrid=True, gridcolor=colors['border'], color=colors['text_secondary']),
        hovermode='x unified',
    )
    return get_plotly_html(fig, 350)

def create_weekly_chart(df, colors):
    df['DiaSemana'] = df['Fecha'].dt.day_name()
    week_counts = df.groupby('DiaSemana').size()
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_labels = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    counts = [week_counts.get(d, 0) for d in day_order]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=day_labels,
        y=counts,
        marker_color=[colors['primary'] if c == max(counts) else colors['secondary'] + '60' for c in counts],
        hovertemplate='<b>%{x}</b><br>Bizums: %{y}<extra></extra>',
    ))
    
    fig.update_layout(
        paper_bgcolor='transparent',
        plot_bgcolor='transparent',
        font=dict(family='Inter, sans-serif', color=colors['text']),
        height=250,
        margin=dict(l=40, r=20, t=20, b=40),
        showlegend=False,
        xaxis=dict(showgrid=False, color=colors['text_secondary']),
        yaxis=dict(showgrid=True, gridcolor=colors['border'], color=colors['text_secondary']),
    )
    return get_plotly_html(fig, 250)

def create_affinity_matrix(df, colors):
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
        paper_bgcolor='transparent',
        plot_bgcolor='transparent',
        font=dict(family='Inter, sans-serif', color=colors['text'], size=10),
        height=350,
        margin=dict(l=60, r=20, t=20, b=40),
        xaxis=dict(tickangle=45, showgrid=False, color=colors['text_secondary']),
        yaxis=dict(showgrid=False, color=colors['text_secondary']),
    )
    return get_plotly_html(fig, 350)

def analyze_data(df):
    if df is None:
        return {}
    
    stats = {}
    stats['total_bizums'] = len(df)
    stats['total_amount'] = df['Cantidad'].sum()
    stats['avg_per_day'] = len(df) / df['Fecha'].nunique()
    stats['avg_per_week'] = len(df) / (df['Fecha'].max() - df['Fecha'].min()).days * 7
    stats['avg_per_month'] = len(df) / (df['Fecha'].max() - df['Fecha'].min()).days * 30
    
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
        days_to_100 = remaining / (model.coef_[0] * (len(friend_df) / days_elapsed)) if days_elapsed > 0 else None
    
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

def dashboard_view(df, stats):
    colors = get_colors()
    
    page.session.set('glass_color', colors['card_glass'])
    page.session.set('border_color', colors['border'])
    
    children = [
        ft.Text('Dashboard', size=32, weight=ft.FontWeight.BOLD, color=colors['text']),
        ft.Text('Resumen de actividad Bizum', size=14, color=colors['text_secondary']),
        ft.Container(height=20),
    ]
    
    children.append(ft.Row([
        stat_card('Total Bizums', str(stats['total_bizums']), f'{stats["avg_per_day"]:.1f} / día', ft.icons.PAYMENTS, 'primary'),
        stat_card('Cantidad Total', f'{stats["total_amount"]:.0f}€', f'{stats["total_amount"]/max(stats["total_bizums"],1):.1f}€ avg', ft.icons.ATTACH_MONEY, 'success'),
    ], spacing=16))
    
    children.append(ft.Container(height=16))
    children.append(ft.Row([
        stat_card('Semanal', f'{stats["avg_per_week"]:.1f}', 'promedio', ft.icons.CALENDAR_VIEW_WEEK, 'secondary'),
        stat_card('Mensual', f'{stats["avg_per_month"]:.1f}', 'promedio', ft.icons.CALENDAR_MONTH, 'warning'),
    ], spacing=16))
    
    children.append(ft.Container(height=24))
    children.append(ft.Text('Top Enviadores', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    
    top_senders = sorted(stats['friend_stats'].items(), key=lambda x: x[1]['sent'], reverse=True)[:3]
    max_count = max([s[1]['sent'] for s in top_senders]) if top_senders else 1
    
    for i, (friend, data) in enumerate(top_senders, 1):
        children.append(ranking_row(friend, i, data['sent'], data['amount_sent'], max_count, True))
        children.append(ft.Container(height=8))
    
    children.append(ft.Container(height=24))
    children.append(ft.Text('Evolución Acumulada', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    children.append(glass_card(ft.Html(create_cumulative_chart(df, colors), allow_row_height_measurement=True), padding=0))
    
    return ft.Column(children, spacing=0, scroll=ft.ScrollMode.AUTO)

def rankings_view(df, stats):
    colors = get_colors()
    
    children = [
        ft.Text('Rankings', size=32, weight=ft.FontWeight.BOLD, color=colors['text']),
        ft.Text('¿Quién manda más Bizums?', size=14, color=colors['text_secondary']),
        ft.Container(height=20),
    ]
    
    children.append(ft.Text('Top 3 Enviadores', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    
    sorted_by_sent = sorted(stats['friend_stats'].items(), key=lambda x: x[1]['sent'], reverse=True)
    max_sent = max([s[1]['sent'] for s in sorted_by_sent]) if sorted_by_sent else 1
    
    for i, (friend, data) in enumerate(sorted_by_sent[:3], 1):
        children.append(ranking_row(friend, i, data['sent'], data['amount_sent'], max_sent, True))
        children.append(ft.Container(height=8))
    
    children.append(ft.Container(height=24))
    children.append(ft.Text('Top 3 Recibidores', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    
    sorted_by_recv = sorted(stats['friend_stats'].items(), key=lambda x: x[1]['received'], reverse=True)
    max_recv = max([s[1]['received'] for s in sorted_by_recv]) if sorted_by_recv else 1
    
    for i, (friend, data) in enumerate(sorted_by_recv[:3], 1):
        children.append(ranking_row(friend, i, data['received'], data['received'] * 20, max_recv, True))
        children.append(ft.Container(height=8))
    
    children.append(ft.Container(height=24))
    children.append(ft.Text('Ranking Global', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    
    sorted_by_total = sorted(stats['friend_stats'].items(), key=lambda x: x[1]['total'], reverse=True)
    max_total = max([s[1]['total'] for s in sorted_by_total]) if sorted_by_total else 1
    
    table_rows = []
    for i, (friend, data) in enumerate(sorted_by_total, 1):
        table_rows.append(
            ft.DataRow(cells=[
                ft.DataCell(ft.Text(f'#{i}', weight=ft.FontWeight.BOLD, color=colors['primary'])),
                ft.DataCell(ft.Text(friend, weight=ft.FontWeight.W_600)),
                ft.DataCell(ft.Text(str(data['sent']))),
                ft.DataCell(ft.Text(str(data['received']))),
                ft.DataCell(ft.Text(f'{data["amount_sent"]:.0f}€')),
            ])
        )
    
    children.append(glass_card(
        ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text('#', color=colors['text_secondary'])),
                ft.DataColumn(ft.Text('Amigo', color=colors['text_secondary'])),
                ft.DataColumn(ft.Text('Enviados', color=colors['text_secondary'])),
                ft.DataColumn(ft.Text('Recibidos', color=colors['text_secondary'])),
                ft.DataColumn(ft.Text('Total €', color=colors['text_secondary'])),
            ],
            rows=table_rows,
        ),
        padding=8,
    ))
    
    if 'year_growth' in stats:
        children.append(ft.Container(height=24))
        children.append(ft.Container(
            padding=16,
            border_radius=12,
            bgcolor=colors['card'],
            content=ft.Row([
                ft.Icon(ft.icons.TRENDING_UP, size=24, color=colors['success'] if stats['year_growth'] > 0 else colors['warning']),
                ft.Column([
                    ft.Text(f'Comparativa {stats["prev_year"]} vs {stats["current_year"]}', size=14, weight=ft.FontWeight.W_600, color=colors['text']),
                    ft.Text(f'{stats["prev_count"]} → {stats["current_count"]} bizums ({stats["year_growth"]:+.1f}%)', 
                           size=12, color=colors['text_secondary']),
                ], expand=True),
            ]),
        ))
    
    return ft.Column(children, spacing=0, scroll=ft.ScrollMode.AUTO)

def trends_view(df, stats):
    colors = get_colors()
    
    children = [
        ft.Text('Tendencias', size=32, weight=ft.FontWeight.BOLD, color=colors['text']),
        ft.Text('Análisis temporal de la actividad', size=14, color=colors['text_secondary']),
        ft.Container(height=20),
    ]
    
    children.append(ft.Text('Mapa de Calor: Día de la semana', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    children.append(glass_card(ft.Html(create_heatmap(df, colors), allow_row_height_measurement=True), padding=0))
    
    children.append(ft.Container(height=24))
    children.append(ft.Text('Actividad por Día de la Semana', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    children.append(glass_card(ft.Html(create_weekly_chart(df, colors), allow_row_height_measurement=True), padding=0))
    
    children.append(ft.Container(height=24))
    children.append(ft.Text('Detección de Picos', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    
    df['FechaStr'] = df['Fecha'].dt.date
    daily_counts = df.groupby('FechaStr').size()
    mean = daily_counts.mean()
    std = daily_counts.std()
    threshold = mean + 1.5 * std
    
    anomalies = daily_counts[daily_counts > threshold]
    
    for date, count in anomalies.items():
        children.append(ft.Container(
            padding=12,
            border_radius=10,
            bgcolor=colors['card'],
            content=ft.Row([
                ft.Container(
                    ft.Icon(ft.icons.WARNING_ROUNDED, size=20, color=colors['warning']),
                    padding=8,
                    border_radius=8,
                    bgcolor=colors['warning'] + '15',
                ),
                ft.Column([
                    ft.Text(f'Pico detectado: {count} bizums', size=14, weight=ft.FontWeight.W_600, color=colors['text']),
                    ft.Text(f'{date}', size=12, color=colors['text_secondary']),
                ], expand=True),
            ]),
        ))
        children.append(ft.Container(height=8))
    
    if len(anomalies) == 0:
        children.append(glass_card(
            ft.Text('No se detectaron picos significativos', color=colors['text_secondary']),
            padding=16,
        ))
    
    return ft.Column(children, spacing=0, scroll=ft.ScrollMode.AUTO)

def forecast_view(df, stats):
    colors = get_colors()
    
    children = [
        ft.Text('Proyecciones', size=32, weight=ft.FontWeight.BOLD, color=colors['text']),
        ft.Text('Predicciones con Machine Learning', size=14, color=colors['text_secondary']),
        ft.Container(height=20),
    ]
    
    children.append(ft.Text('Evolución y Proyección', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    children.append(glass_card(ft.Html(create_forecast_chart(df, colors), allow_row_height_measurement=True), padding=0))
    
    children.append(ft.Container(height=24))
    children.append(ft.Text('Predicciones por Persona (Fin de Año)', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    
    forecast_data = []
    for friend in FRIENDS:
        if friend in df['Emisor'].values:
            fc = get_forecast(df, friend)
            if fc:
                forecast_data.append((friend, fc))
    
    forecast_data.sort(key=lambda x: x[1]['predicted'], reverse=True)
    
    for friend, fc in forecast_data[:6]:
        progress = min(fc['predicted'] / 100, 1.0)
        children.append(ft.Container(
            padding=16,
            border_radius=12,
            bgcolor=colors['card'],
            content=ft.Column([
                ft.Row([
                    ft.Text(friend, size=16, weight=ft.FontWeight.W_600, color=colors['text']),
                    ft.Text(f'{fc["current"]} → {fc["predicted"]}', size=14, color=colors['primary'], weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(height=8),
                ft.Stack([
                    ft.Container(
                        height=8,
                        border_radius=4,
                        bgcolor=colors['bg'],
                    ),
                    ft.Container(
                        height=8,
                        width=max(fc['current'] / 100 * 200, 4),
                        border_radius=4,
                        gradient=ft.LinearGradient(colors=[colors['primary'], colors['secondary']]),
                    ),
                ]),
                ft.Container(height=4),
                ft.Row([
                    ft.Text(f'{(progress*100):.0f}% del objetivo', size=11, color=colors['text_secondary']),
                    ft.Text(f'~{fc["daily_rate"]:.1f}/día', size=11, color=colors['text_secondary']),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ]),
        ))
        children.append(ft.Container(height=8))
    
    children.append(ft.Container(height=24))
    children.append(ft.Text('Hitos - ¿Quién alcanza 100 primero?', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    
    milestones = []
    for friend, fc in forecast_data:
        if fc['days_to_100'] and fc['days_to_100'] > 0 and fc['days_to_100'] < 365:
            milestones.append((friend, fc['days_to_100']))
    
    milestones.sort(key=lambda x: x[1])
    
    if milestones:
        for i, (friend, days) in enumerate(milestones[:3], 1):
            target_date = (datetime.now() + timedelta(days=int(days))).strftime('%d %b %Y')
            children.append(ft.Container(
                padding=16,
                border_radius=12,
                bgcolor=colors['card'],
                content=ft.Row([
                    ft.Container(
                        ft.Text(f'#{i}', size=20, weight=ft.FontWeight.BOLD, color=colors['success']),
                        padding=10,
                        border_radius=10,
                        bgcolor=colors['success'] + '15',
                    ),
                    ft.Column([
                        ft.Text(friend, size=16, weight=ft.FontWeight.W_600, color=colors['text']),
                        ft.Text(f'~{int(days)} días → {target_date}', size=12, color=colors['text_secondary']),
                    ], expand=True),
                    ft.Icon(ft.icons.EMOJI_EVENTS, color=colors['warning']),
                ]),
            ))
            children.append(ft.Container(height=8))
    else:
        children.append(glass_card(
            ft.Text('No se prevén hitos de 100 bizums este año', color=colors['text_secondary']),
            padding=16,
        ))
    
    return ft.Column(children, spacing=0, scroll=ft.ScrollMode.AUTO)

def probability_view(df, stats):
    colors = get_colors()
    
    children = [
        ft.Text('Probabilidades', size=32, weight=ft.FontWeight.BOLD, color=colors['text']),
        ft.Text('Análisis probabilístico del comportamiento', size=14, color=colors['text_secondary']),
        ft.Container(height=20),
    ]
    
    children.append(ft.Text('Probabilidad Condicional por Día', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    probs_data = []
    for friend in ['Tom', 'Edu', 'Victor', 'Cesar', 'Quique']:
        if friend in df['Emisor'].values:
            probs = [get_conditional_prob(df, friend, day) for day in weekdays]
            probs_data.append((friend, probs))
    
    if probs_data:
        for friend, probs in probs_data[:3]:
            children.append(ft.Container(
                padding=12,
                border_radius=12,
                bgcolor=colors['card'],
                content=ft.Column([
                    ft.Text(friend, size=14, weight=ft.FontWeight.W_600, color=colors['text']),
                    ft.Container(height=8),
                    ft.Row([
                        ft.Column([
                            ft.Text(name, size=10, color=colors['text_secondary']),
                            ft.Text(f'{p:.1f}%', size=12, weight=ft.FontWeight.BOLD, 
                                   color=colors['primary'] if p > 20 else colors['text_secondary']),
                        ]) for name, p in zip(weekday_names, probs)
                    ], spacing=8, scroll=ft.ScrollMode.AUTO),
                ]),
            ))
            children.append(ft.Container(height=8))
    
    children.append(ft.Container(height=24))
    children.append(ft.Text('Matriz de Afinidad', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Text('¿Quién interactúa más con quién?', size=12, color=colors['text_secondary']))
    children.append(ft.Container(height=12))
    
    affinity_html = create_affinity_matrix(df, colors)
    if affinity_html:
        children.append(glass_card(ft.Html(affinity_html, allow_row_height_measurement=True), padding=0))
    
    children.append(ft.Container(height=24))
    children.append(ft.Text('Análisis de Patrones', size=18, weight=ft.FontWeight.W_600, color=colors['text']))
    children.append(ft.Container(height=12))
    
    if stats['weekday_probs']:
        max_day = max(stats['weekday_probs'], key=stats['weekday_probs'].get)
        day_translations = {
            'Monday': 'lunes', 'Tuesday': 'martes', 'Wednesday': 'miércoles',
            'Thursday': 'jueves', 'Friday': 'viernes', 'Saturday': 'sábado', 'Sunday': 'domingo'
        }
        
        children.append(glass_card(
            ft.Column([
                ft.Row([
                    ft.Icon(ft.icons.STAR_ROUNDED, size=24, color=colors['warning']),
                    ft.Column([
                        ft.Text(f'El {day_translations.get(max_day, max_day)} es el día más activo', 
                               size=14, weight=ft.FontWeight.W_600, color=colors['text']),
                        ft.Text(f'{stats["weekday_probs"][max_day]} bizums registrados', 
                               size=12, color=colors['text_secondary']),
                    ], expand=True),
                ]),
            ]),
            padding=16,
        ))
    
    return ft.Column(children, spacing=0, scroll=ft.ScrollMode.AUTO)

def create_tab(icon, selected_icon, label, view_id, is_selected):
    colors = get_colors()
    return ft.Container(
        content=ft.Column([
            ft.Icon(selected_icon if is_selected else icon, size=24, 
                   color=colors['primary'] if is_selected else colors['text_secondary']),
            ft.Text(label, size=10, color=colors['primary'] if is_selected else colors['text_secondary'],
                   weight=ft.FontWeight.W_500 if is_selected else ft.FontWeight.NORMAL),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
        padding=ft.padding.symmetric(horizontal=16, vertical=8),
        border_radius=16,
        bgcolor=colors['primary'] + '15' if is_selected else 'transparent',
        on_click=lambda e: switch_view(view_id),
    )

def switch_view(view_id):
    page.session.set('current_view', view_id)
    page.update()

def main(page: ft.Page):
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window_width = 420
    page.window_height = 750
    page.window_resizable = True
    page.title = 'Bizum Analytics'
    
    page.session.set('current_view', 'dashboard')
    
    df = load_data()
    stats = analyze_data(df)
    
    def update_theme(e):
        page.theme_mode = ft.ThemeMode.DARK if page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        page.update()
    
    def get_nav():
        colors = get_colors()
        current = page.session.get('current_view')
        return ft.Container(
            content=ft.Row([
                create_tab(ft.icons.HOME_OUTLINED, ft.icons.HOME, 'Inicio', 'dashboard', current == 'dashboard'),
                create_tab(ft.icons.LEADERBOARD_OUTLINED, ft.icons.LEADERBOARD, 'Rankings', 'rankings', current == 'rankings'),
                create_tab(ft.icons.TRENDING_UP_OUTLINED, ft.icons.TRENDING_UP, 'Tendencias', 'trends', current == 'trends'),
                create_tab(ft.icons.INSIGHTS_OUTLINED, ft.icons.INSIGHTS, 'Forecast', 'forecast', current == 'forecast'),
                create_tab(ft.icons.CALCULATE_OUTLINED, ft.icons.CALCULATE, 'Probabilidades', 'probability', current == 'probability'),
            ], spacing=0, alignment=ft.MainAxisAlignment.SPACE_EVENLY),
            padding=ft.padding.symmetric(horizontal=8, vertical=8),
            border_radius=20,
            bgcolor=colors['card'],
            border=ft.border.all(1, colors['border']),
        )
    
    def get_content():
        current = page.session.get('current_view')
        colors = get_colors()
        
        if current == 'dashboard':
            content = dashboard_view(df, stats)
        elif current == 'rankings':
            content = rankings_view(df, stats)
        elif current == 'trends':
            content = trends_view(df, stats)
        elif current == 'forecast':
            content = forecast_view(df, stats)
        elif current == 'probability':
            content = probability_view(df, stats)
        else:
            content = dashboard_view(df, stats)
        
        return ft.Container(
            content=ft.Column([
                ft.Container(height=60),
                content,
                ft.Container(height=100),
            ], spacing=0),
            padding=ft.padding.symmetric(horizontal=20),
        )
    
    def get_header():
        colors = get_colors()
        return ft.Container(
            content=ft.Row([
                ft.Column([
                    ft.Text('Bizum Analytics', size=20, weight=ft.FontWeight.BOLD, color=colors['text']),
                ], expand=True),
                ft.Container(
                    ft.Icon(ft.icons.DARK_MODE_OUTLINED if page.theme_mode == ft.ThemeMode.LIGHT else ft.icons.LIGHT_MODE_OUTLINED, 
                           size=24, color=colors['text_secondary']),
                    on_click=update_theme,
                ),
            ]),
            padding=ft.padding.symmetric(horizontal=20, vertical=12),
        )
    
    def main_container(e=None):
        colors = get_colors()
        page.controls.clear()
        
        page.add(
            ft.Container(
                content=ft.Column([
                    get_header(),
                    ft.Container(
                        content=get_content(),
                        expand=True,
                        scroll=ft.ScrollMode.AUTO,
                    ),
                    ft.Container(
                        content=get_nav(),
                        padding=ft.padding.only(bottom=20, left=20, right=20),
                    ),
                ], spacing=0),
                bgcolor=colors['bg'],
            )
        )
        
        if e:
            page.update()
    
    page.on_resize = main_container
    main_container()

if __name__ == '__main__':
    ft.app(target=main)
