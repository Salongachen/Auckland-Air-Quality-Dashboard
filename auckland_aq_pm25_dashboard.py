# auckland_aq_pm25_dashboard.py
"""
Auckland Council — PM2.5 Exceedance & Comparison Dashboard
Single-file Streamlit app (multi-site uploads). Includes:
- Robust cleaning + PM2.5 detection
- Daily 24-hr averaging + NESAQ exceedance (25 µg/m3)
- Multi-site comparison: time series, yearly exceedances, monthly seasonality
- Forecasting (Prophet optional, SARIMAX fallback) — user-run
- RandomForest exceedance classifier with TimeSeriesSplit CV
- Technical Tab with model evaluation and downloadable PDF report per site
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, os, tempfile
from datetime import timedelta
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

st.set_page_config(page_title="Auckland AQ — PM2.5 Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("Auckland Council — PM2.5 Exceedance & Comparison Dashboard")

# --------------------------
# Configuration
# --------------------------
NESAQ_PM25 = 25.0  # 24-hr μg/m³ threshold
SAMPLE_DATA_BTN = False  # set True to provide sample generator button

# --------------------------
# Utility & Cleaning
# --------------------------
@st.cache_data
def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif name.endswith('.xlsx') or name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            st.error("Unsupported file type: " + uploaded_file.name)
            return None
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Failed to read {uploaded_file.name}: {e}")
        return None

def standardize_param(s):
    if pd.isna(s):
        return s
    s = str(s).strip().lower()
    s = s.replace(' ', '').replace('-', '').replace('_','')
    return s

def detect_pm25_rows(df):
    if 'Parameter' in df.columns:
        param_series = df['Parameter'].astype(str).apply(standardize_param)
        mask = param_series.str.contains(r'pm25|pm2\.5|pm2', regex=True, na=False)
        return mask
    else:
        # no Parameter column -> assume single pollutant file (treat as PM2.5)
        return pd.Series([True]*len(df), index=df.index)

def clean_value(x):
    try:
        if isinstance(x, str):
            s = x.replace(',', '').strip()
            if s == '' or s in ['--','NA','N/A','na','-']:
                return np.nan
            if s.startswith('<'):
                s = s.lstrip('<').strip()
            return float(s)
        return float(x)
    except Exception:
        return np.nan

def infer_datetime(df):
    cols = [c.lower() for c in df.columns]
    # explicit datetime column
    if 'datetime' in cols:
        c = df.columns[cols.index('datetime')]
        df['Datetime'] = pd.to_datetime(df[c], errors='coerce')
        return df
    # try date + time
    date_col = None; time_col = None
    for cand in ['date','day','sample_date','measurement_date']:
        if cand in cols:
            date_col = df.columns[cols.index(cand)]; break
    for cand in ['time','hour','sample_time','measurement_time']:
        if cand in cols:
            time_col = df.columns[cols.index(cand)]; break
    if date_col:
        if time_col:
            df['Datetime'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), errors='coerce')
        else:
            df['Datetime'] = pd.to_datetime(df[date_col], errors='coerce')
        return df
    # fallback: detect any column that looks like datetimes
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(8).tolist()
        parsed = 0
        for s in sample:
            try:
                pd.to_datetime(s)
                parsed += 1
            except Exception:
                pass
        if parsed >= max(3, len(sample)//2):
            df['Datetime'] = pd.to_datetime(df[c], errors='coerce')
            return df
    return df

def prepare_daily_series(df, site_name=None, require_pm25=True, allow_aqi_fallback=False):
    df = df.copy()
    df = infer_datetime(df)
    if 'Datetime' not in df.columns:
        return None, "No Datetime detected. Please ensure file has Date/Time or Datetime columns."
    if 'Site' not in df.columns:
        df['Site'] = site_name if site_name else 'Site_Unknown'
    if 'Parameter' in df.columns:
        df['Parameter_std'] = df['Parameter'].astype(str).apply(standardize_param)
    else:
        df['Parameter_std'] = ''
    pm_mask = detect_pm25_rows(df)
    if pm_mask.sum() == 0:
        if allow_aqi_fallback and 'Parameter' in df.columns and df['Parameter'].astype(str).str.contains('AQI', case=False, na=False).any():
            # warn and treat Value as PM2.5 proxy
            df['Parameter_std'] = 'pm25'
            pm_mask = pd.Series([True]*len(df), index=df.index)
        elif require_pm25:
            return None, "No PM2.5 records detected in this file. Enable AQI fallback or upload PM2.5 data."
        else:
            pm_mask = pd.Series([True]*len(df), index=df.index)

    pm_df = df[pm_mask].copy()
    if 'Value' not in pm_df.columns:
        return None, "No 'Value' column found in file."
    pm_df['Value_clean'] = pm_df['Value'].apply(clean_value)
    pm_df = pm_df.dropna(subset=['Datetime','Value_clean']).copy()
    if pm_df.empty:
        return None, "After cleaning, no valid PM2.5 measurements remain."
    pm_df['Datetime'] = pd.to_datetime(pm_df['Datetime'], errors='coerce')
    pm_df = pm_df.dropna(subset=['Datetime'])
    pm_df['Date'] = pm_df['Datetime'].dt.floor('D')
    daily = pm_df.groupby(['Site','Date']).agg(
        PM25_daily_avg=('Value_clean','mean'),
        records=('Value_clean','count')
    ).reset_index().sort_values(['Site','Date'])
    daily['Exceedance'] = daily['PM25_daily_avg'] > NESAQ_PM25
    return daily, None

# --------------------------
# Sidebar: upload and settings
# --------------------------
st.sidebar.header("Upload & Settings")
uploaded_files = st.sidebar.file_uploader("Upload CSV or Excel files (one file per site). Multiple selection allowed.", accept_multiple_files=True, type=['csv','xlsx','xls'])
require_pm25 = st.sidebar.checkbox("Require PM2.5 in uploads (warn if missing)", value=True)
allow_aqi_fallback = st.sidebar.checkbox("Allow AQI fallback (treat AQI values as PM2.5 proxy)", value=False)
st.sidebar.markdown("**Model & Evaluation settings**")
n_lags = st.sidebar.number_input("Lag days (for classifier)", min_value=1, max_value=21, value=7)
cv_splits = st.sidebar.number_input("Classifier CV splits (TimeSeriesSplit)", min_value=2, max_value=10, value=5)
forecast_horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=7, max_value=90, value=30)
forecast_backtest_folds = st.sidebar.number_input("Forecast backtest folds", min_value=1, max_value=5, value=3)

# Optional sample button
if SAMPLE_DATA_BTN:
    if st.sidebar.button("Load demo sample"):
        # generate small synthetic dataset if desired
        pass

if not uploaded_files or len(uploaded_files) == 0:
    st.info("Upload one or more site files (CSV or Excel). Each file is treated as a site (default site name = filename).")
    st.stop()

# --------------------------
# Process files
# --------------------------
site_daily_list = []
site_names = []
processing_msg = st.sidebar.empty()
i = 0
for uploaded in uploaded_files:
    i += 1
    processing_msg.text(f"Processing {i}/{len(uploaded_files)}: {uploaded.name}")
    df = read_uploaded_file(uploaded)
    if df is None:
        st.warning(f"Skipping {uploaded.name} (read error).")
        continue
    site_name = os.path.splitext(uploaded.name)[0]
    daily_df, err = prepare_daily_series(df, site_name=site_name, require_pm25=require_pm25, allow_aqi_fallback=allow_aqi_fallback)
    if daily_df is None:
        st.error(f"File {uploaded.name}: {err}")
        continue
    site_daily_list.append(daily_df)
    site_names.append(daily_df['Site'].iloc[0])
processing_msg.text("Processing complete.")

if not site_daily_list:
    st.error("No valid PM2.5 data found in uploaded files.")
    st.stop()

all_sites_daily = pd.concat(site_daily_list, ignore_index=True)
all_sites_daily['Date'] = pd.to_datetime(all_sites_daily['Date'])

# --------------------------
# Quick summary & downloads
# --------------------------
st.header("Loaded sites & summary")
sites_sorted = sorted(set(site_names))
col1, col2 = st.columns([3,1])
with col1:
    st.write("Sites:", ", ".join(sites_sorted))
    summary_rows = []
    for s in sites_sorted:
        d = all_sites_daily[all_sites_daily['Site']==s]
        summary_rows.append({
            'Site': s,
            'Start': d['Date'].min().date(),
            'End': d['Date'].max().date(),
            'Days': d['Date'].nunique(),
            'ExceedDays': int(d['Exceedance'].sum())
        })
    st.dataframe(pd.DataFrame(summary_rows).sort_values('Site').reset_index(drop=True), use_container_width=True)
with col2:
    # download cleaned daily csv for all sites
    buf = io.StringIO()
    all_sites_daily.to_csv(buf, index=False)
    st.download_button("Download cleaned daily CSV (all sites)", data=buf.getvalue(), file_name="daily_pm25_all_sites.csv", mime="text/csv")

# --------------------------
# Tabs: Council view + Technical tab
# --------------------------
tabs = st.tabs(["Exceedance & Trends", "Forecast & Alerts", "Technical (Advanced)"])

# --- Tab 1: Exceedance & Trends ---
with tabs[0]:
    st.subheader("Time series & exceedances")
    selected_sites = st.multiselect("Select site(s) to visualize", options=sites_sorted, default=[sites_sorted[0]])
    sel = all_sites_daily[all_sites_daily['Site'].isin(selected_sites)].copy()
    # time series
    fig_ts = px.line(sel, x='Date', y='PM25_daily_avg', color='Site', title='Daily PM2.5 (24-hr avg)')
    fig_ts.add_hline(y=NESAQ_PM25, line_dash='dash', annotation_text=f'NESAQ {NESAQ_PM25} μg/m³', annotation_position='top left')
    st.plotly_chart(fig_ts, use_container_width=True)
    # yearly exceedance counts
    st.subheader("Exceedance days per year")
    sel['Year'] = sel['Date'].dt.year
    ex_by_year = sel.groupby(['Site','Year'])['Exceedance'].sum().reset_index()
    fig_year = px.bar(ex_by_year, x='Year', y='Exceedance', color='Site', barmode='group', title='Exceedance days per year')
    st.plotly_chart(fig_year, use_container_width=True)
    # monthly seasonality
    st.subheader("Monthly average (seasonality)")
    sel['Month'] = sel['Date'].dt.month
    monthly_avg = sel.groupby(['Site','Month'])['PM25_daily_avg'].mean().reset_index()
    fig_month = px.line(monthly_avg, x='Month', y='PM25_daily_avg', color='Site', markers=True, title='Monthly average PM2.5 (daily averages)')
    st.plotly_chart(fig_month, use_container_width=True)
    # table toggle
    if st.checkbox("Show cleaned daily data table"):
        st.dataframe(sel.sort_values(['Site','Date']).reset_index(drop=True), use_container_width=True)

# --- Tab 2: Forecast & Alerts ---
with tabs[1]:
    st.subheader("Forecasting & Predicted Exceedances")
    st.info("Run forecasts per selected site. Forecast uses Prophet if installed; else SARIMAX fallback is used.")
    # detect Prophet
    try:
        from prophet import Prophet
        prophet_ok = True
    except Exception:
        prophet_ok = False
    st.write("Prophet installed:" , "Yes" if prophet_ok else "No (using SARIMAX fallback)")
    forecast_sites = st.multiselect("Select site(s) to forecast", options=sites_sorted, default=[sites_sorted[0]])
    fh = int(st.number_input("Forecast horizon (days)", min_value=7, max_value=365, value=int(forecast_horizon)))
    if st.button("Run forecasts (per selected site)"):
        for site in forecast_sites:
            st.markdown(f"### Forecast — {site}")
            site_df = all_sites_daily[all_sites_daily['Site']==site][['Date','PM25_daily_avg']].rename(columns={'Date':'ds','PM25_daily_avg':'y'}).sort_values('ds')
            if len(site_df) < 30:
                st.warning(f"Site {site} has less than 30 days — forecast may be unreliable.")
            if prophet_ok:
                try:
                    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
                    m.fit(site_df)
                    future = m.make_future_dataframe(periods=fh)
                    fc = m.predict(future)
                    fig = px.line(fc, x='ds', y='yhat', title=f'{site} — Forecast (Prophet)')
                    fig.add_traces(px.line(fc, x='ds', y='yhat_lower').data)
                    fig.add_traces(px.line(fc, x='ds', y='yhat_upper').data)
                    fig.add_hline(y=NESAQ_PM25, line_dash='dash', annotation_text=f'NESAQ {NESAQ_PM25}', annotation_position='top left')
                    st.plotly_chart(fig, use_container_width=True)
                    # predicted exceedances in horizon
                    future_only = fc[fc['ds'] > site_df['ds'].max()]
                    pred_exc = future_only[future_only['yhat'] > NESAQ_PM25][['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'ds':'Date','yhat':'Predicted_PM25'})
                    if not pred_exc.empty:
                        st.markdown(f"**Predicted exceedances for {site}:**")
                        st.dataframe(pred_exc)
                    else:
                        st.info(f"No predicted exceedances for {site} in next {fh} days.")
                except Exception as e:
                    st.error(f"Prophet forecast error for {site}: {e}")
            else:
                # SARIMAX fallback
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    ts = site_df.set_index('ds')['y'].asfreq('D').interpolate(limit=3)
                    model = SARIMAX(ts, order=(1,0,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    pred = res.get_forecast(steps=fh)
                    pred_mean = pred.predicted_mean
                    pred_ci = pred.conf_int()
                    fc = pd.DataFrame({'ds': pred_mean.index, 'yhat': pred_mean.values, 'yhat_lower': pred_ci.iloc[:,0].values, 'yhat_upper': pred_ci.iloc[:,1].values})
                    fig = px.line(fc, x='ds', y='yhat', title=f'{site} — Forecast (SARIMAX fallback)')
                    fig.add_hline(y=NESAQ_PM25, line_dash='dash', annotation_text=f'NESAQ {NESAQ_PM25}', annotation_position='top left')
                    st.plotly_chart(fig, use_container_width=True)
                    pred_exc = fc[fc['yhat'] > NESAQ_PM25]
                    if not pred_exc.empty:
                        st.markdown(f"**Predicted exceedances for {site}:**")
                        st.dataframe(pred_exc[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'ds':'Date','yhat':'Predicted_PM25'}))
                    else:
                        st.info(f"No predicted exceedances for {site} in next {fh} days (SARIMAX).")
                except Exception as e:
                    st.error(f"SARIMAX forecast failed for {site}: {e}")

# --- Tab 3: Technical (Advanced) ---
with tabs[2]:
    st.subheader("Technical — Model Evaluation & Classifier Backtesting")
    st.info("This tab is intended for data scientists and technical reviewers. It contains classifier CV, forecast backtesting, feature importances and downloadable technical PDF reports.")
    # classifier controls
    run_classifier = st.button("Run RandomForest exceedance classifier CV")
    run_backtest = st.button("Run forecast backtesting")
    # functions for classifier/backtest
    def create_lag_features(site_df, lags=7):
        df = site_df[['Date','PM25_daily_avg','Exceedance']].sort_values('Date').reset_index(drop=True).copy()
        for lag in range(1, lags+1):
            df[f'lag_{lag}'] = df['PM25_daily_avg'].shift(lag)
        df['dow'] = df['Date'].dt.dayofweek
        df['target_exceed_tplus1'] = df['Exceedance'].shift(-1).astype(float)
        df_model = df.dropna().reset_index(drop=True)
        feature_cols = [f'lag_{l}' for l in range(1, lags+1)] + ['dow']
        return df_model, feature_cols, 'target_exceed_tplus1'
    def run_rf_cv(df_model, feature_cols, target_col, n_splits=5):
        X = df_model[feature_cols].values
        y = df_model[target_col].values.astype(int)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            results.append({
                'fold': fold,
                'precision': precision_score(y_te, y_pred, zero_division=0),
                'recall': recall_score(y_te, y_pred, zero_division=0),
                'f1': f1_score(y_te, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_te, y_pred),
                'confusion_matrix': confusion_matrix(y_te, y_pred)
            })
        agg = {k: float(np.mean([r[k] for r in results])) for k in ['precision','recall','f1','accuracy']}
        return results, agg
    def train_final_rf(df_model, feature_cols, target_col):
        X = df_model[feature_cols].values
        y = df_model[target_col].values.astype(int)
        pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))])
        pipe.fit(X, y)
        return pipe
    def forecast_backtest(site_df, horizon=30, folds=3):
        series = site_df[['Date','PM25_daily_avg']].sort_values('Date').reset_index(drop=True)
        total = len(series)
        if total < (horizon + 20):
            return None, "Insufficient daily data for reliable backtest."
        train_end = total - horizon * folds
        if train_end < 30:
            train_end = int(total * 0.6)
        try:
            from prophet import Prophet
            use_prophet = True
        except Exception:
            use_prophet = False
        res_list = []
        for f in range(folds):
            train = series.iloc[:train_end + f*horizon].rename(columns={'Date':'ds','PM25_daily_avg':'y'})[['ds','y']].copy()
            test = series.iloc[train_end + f*horizon : train_end + (f+1)*horizon].copy()
            if use_prophet:
                m = Prophet(daily_seasonality=True, yearly_seasonality=True)
                m.fit(train)
                future = m.make_future_dataframe(periods=horizon)
                fc = m.predict(future)
                y_pred = fc[['ds','yhat']].tail(horizon)['yhat'].values[:len(test)]
            else:
                try:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    ts = train.set_index('ds')['y'].asfreq('D').interpolate(limit=3)
                    model = SARIMAX(ts, order=(1,0,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
                    r = model.fit(disp=False)
                    p = r.get_forecast(steps=horizon)
                    y_pred = p.predicted_mean.values[:len(test)]
                except Exception as e:
                    return None, f"SARIMAX error: {e}"
            y_true = test['PM25_daily_avg'].values
            mae = mean_absolute_error(y_true, y_pred)
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            res_list.append({'fold': f+1, 'mae': mae, 'rmse': rmse, 'n_test': len(y_true)})
        return res_list, None

    # perform classifier CV
    if run_classifier:
        for site in sites_sorted:
            st.markdown(f"#### Classifier CV — {site}")
            site_df = all_sites_daily[all_sites_daily['Site']==site][['Date','PM25_daily_avg','Exceedance']].dropna().sort_values('Date')
            if len(site_df) < (n_lags + 10):
                st.warning(f"{site} — not enough daily rows for classifier (need >= {n_lags+10}, got {len(site_df)}).")
                continue
            df_model, feature_cols, target_col = create_lag_features(site_df, lags=n_lags)
            if df_model.empty:
                st.warning(f"{site} — no rows after lagging.")
                continue
            folds, agg = run_rf_cv(df_model, feature_cols, target_col, n_splits=cv_splits)
            st.write("Per-fold metrics:")
            st.dataframe(pd.DataFrame([{'fold':f['fold'],'precision':f['precision'],'recall':f['recall'],'f1':f['f1'],'accuracy':f['accuracy']} for f in folds]))
            st.write("Aggregate averages:")
            st.json(agg)
            # final model & feature importances
            pipe = train_final_rf(df_model, feature_cols, target_col)
            rf = pipe.named_steps['rf']
            feat_imp = pd.DataFrame({'feature':feature_cols,'importance': rf.feature_importances_}).sort_values('importance', ascending=False)
            st.subheader("Feature importances")
            st.dataframe(feat_imp)

    # forecast backtesting
    if run_backtest:
        for site in sites_sorted:
            st.markdown(f"#### Forecast backtest — {site}")
            site_df = all_sites_daily[all_sites_daily['Site']==site][['Date','PM25_daily_avg']].dropna().sort_values('Date')
            res, err = forecast_backtest(site_df, horizon=int(forecast_horizon), folds=int(forecast_backtest_folds))
            if err:
                st.error(f"{site}: {err}")
                continue
            st.dataframe(pd.DataFrame(res))
            st.write("Average MAE:", float(np.mean([r['mae'] for r in res])), "Average RMSE:", float(np.mean([r['rmse'] for r in res])))

    # PDFs per site
    st.subheader("Download technical PDF report (per site)")
    def build_pdf(site_df, site_name):
        try:
            from fpdf import FPDF
            import matplotlib.pyplot as plt
        except Exception:
            st.error("PDF requires 'fpdf' and 'matplotlib' installed on server.")
            return None
        # snapshot plot
        snapshot = site_df.tail(180)
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(snapshot['Date'], snapshot['PM25_daily_avg'], linewidth=0.8)
        ax.axhline(NESAQ_PM25, linestyle='--', color='red')
        ax.set_title(f'{site_name} — Daily PM2.5 (24-hr avg)')
        ax.set_xlabel('Date'); ax.set_ylabel('μg/m³')
        plt.tight_layout()
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig.savefig(tmp.name, dpi=150)
        plt.close(fig)
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 8, f'Auckland Council — PM2.5 Technical Report ({site_name})', ln=True)
        pdf.ln(3)
        pdf.set_font('Arial','',11)
        pdf.cell(0, 6, f"Date range: {site_df['Date'].min().date()} to {site_df['Date'].max().date()}", ln=True)
        pdf.cell(0, 6, f"Total days: {site_df['Date'].nunique()}", ln=True)
        pdf.cell(0, 6, f"Exceedance days: {int(site_df['Exceedance'].sum())}", ln=True)
        pdf.ln(6)
        pdf.image(tmp.name, x=15, y=60, w=180)
        out = pdf.output(dest='S').encode('latin-1')
        try: os.unlink(tmp.name)
        except: pass
        return out

    for site in sites_sorted:
        st.markdown(f"**{site}**")
        site_df = all_sites_daily[all_sites_daily['Site']==site].sort_values('Date')
        csv_buf = io.StringIO()
        site_df.to_csv(csv_buf, index=False)
        st.download_button(f"Download {site} — daily CSV", data=csv_buf.getvalue(), file_name=f"daily_pm25_{site}.csv", mime="text/csv", key=f"csv_{site}")
        pdf_bytes = build_pdf(site_df, site)
        if pdf_bytes:
            st.download_button(f"Download {site} — technical PDF", data=pdf_bytes, file_name=f"pm25_technical_{site}.pdf", mime="application/pdf", key=f"pdf_{site}")
        else:
            st.info("PDF generator not available (requires fpdf & matplotlib).")

st.markdown("---")
st.info("Notes: This app focuses on PM2.5 exceedance (NESAQ 24-hr = 25 μg/m³). Use the Technical tab for model validation and to download technical PDFs for council records.")
