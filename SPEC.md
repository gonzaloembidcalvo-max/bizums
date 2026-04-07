# Bizum Analytics - iOS 26 Style Desktop App

## Project Overview
- **Name**: Bizum Analytics
- **Type**: Desktop Analytics Application
- **Core Functionality**: Read, analyze and visualize historical Bizum transactions between friends with modern iOS 26 aesthetic
- **Target Users**: Group of friends tracking their Bizum activity (Tom, Edu, Victor, Cesar, Quique, Pablo, Jorge, Tomi, Gonza, Alonso, Dias, Alex)

## Design Specification

### Visual Style - iOS 26 / macOS Sequoia Inspired
- **Background**: Soft gradient backgrounds (#F5F5F7 light / #1C1C1E dark)
- **Cards**: Glassmorphism effect with blur, white/dark overlay (0.7 opacity)
- **Border Radius**: 20px for cards, 12px for buttons, 8px for inputs
- **Shadows**: Soft drop shadows (0 4px 20px rgba(0,0,0,0.08))
- **Typography**: SF Pro / Inter font family, clean and minimal

### Color Palette
- **Primary**: #007AFF (iOS Blue)
- **Secondary**: #5856D6 (Purple)
- **Success**: #34C759 (Green)
- **Warning**: #FF9500 (Orange)
- **Destructive**: #FF3B30 (Red)
- **Accent Gradients**: Blue→Purple for highlights

### Navigation
- **Type**: Bottom tab bar with 5 sections
- **Tabs**: Dashboard, Rankings, Trends, Forecast, Probabilities
- **Icons**: Flet icons, selected state with filled icon
- **Style**: Floating pill design with glassmorphism

### Layout Structure
1. **Header**: App title, theme toggle, settings
2. **Content Area**: Scrollable content with cards
3. **Bottom Navigation**: Fixed tab bar

## Features Specification

### 1. Dashboard View
- Summary cards (total bizums, total amount, avg per day)
- Quick stats for each friend
- Recent activity feed

### 2. Rankings View
- Top 3 / Bottom 3 rankings
- Sortable table by bizums sent, received, amount
- Visual progress bars

### 3. Trends View
- Line chart: Cumulative bizums over time (smooth curves)
- Heatmap: Day of week vs Month activity
- Anomaly detection highlights

### 4. Forecast View
- Linear regression projection
- Year-end prediction per person
- Milestone tracker (100 bizums threshold)
- Progress indicators

### 5. Probability View
- Conditional probability calculator
- Affinity matrix (who interacts with whom)
- Day-of-week probability chart

## Technical Specification

### Libraries
- **Flet**: UI framework (modern, clean)
- **Pandas**: Data manipulation
- **Plotly**: Interactive charts
- **Scikit-learn**: Linear regression
- **Statsmodels**: ARIMA for time series

### Data Model (CSV)
```
Fecha,Emisor,Receptor,Cantidad,Acumulado
2025-01-15,Tom,Edu,15.50,15.50
...
```

### Analysis Modules
1. **DescriptiveStats**: Means, totals, rankings
2. **TemporalAnalysis**: Time series, trends, anomalies
3. **Forecasting**: Linear regression, ARIMA
4. **ProbabilityEngine**: Conditional prob, affinity matrix
