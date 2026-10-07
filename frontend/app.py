import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Superconductivity Tc Predictor", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# Initialize Session State for History
if 'history' not in st.session_state:
    st.session_state.history = []

# Custom CSS for glowing UI
st.markdown("""
    <style>
    .main { background-color: #050810; color: #e2e8f0; }
    .stMetric {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.4));
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 20px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom: 3px solid #38bdf8 !important; }
    </style>
""", unsafe_allow_html=True)

# 2. Sidebar
with st.sidebar:
    st.markdown("## ⚡ Core AI Engine")
    st.markdown("---")
    api_url = st.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000")
    
    if st.button("🔄 Ping Backend Server"):
        try:
            r = requests.get(f"{api_url}/health")
            st.success("API is Online! 🟢") if r.status_code == 200 else st.error("API Error 🔴")
        except:
            st.error("Server Unreachable 🔴")

    st.markdown("---")
    st.markdown("### 🧪 Material Presets")
    preset_choice = st.radio("Configuration:", ["Default Baseline (0.5)", "Low Energy State (0.1)", "High Density Alloy (1.8)"])
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []

# 3. Header
st.title("⚡ Superconductor Critical Temperature ($T_c$)")
st.markdown("Advanced MLOps dashboard with real-time visualization and telemetry.")

# Fetch Features
@st.cache_data(ttl=60)
def fetch_features(base_url):
    try:
        res = requests.get(f"{base_url}/features")
        return res.json().get("expected_features", []) if res.status_code == 200 else []
    except:
        return []

expected_features = fetch_features(api_url)

if not expected_features:
    st.error(f"⚠️ **Connection Lost!** Start FastAPI server at {api_url}.")
else:
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Features", f"{len(expected_features)} Matrix", "Validated")
    col2.metric("Prediction Engine", "XGBoost Regressor", "v2.0")
    col3.metric("API Status", "Connected 🟢", "Port 8000")
    st.markdown("<br>", unsafe_allow_html=True)

    # Multiplier Logic
    multiplier = 0.5
    if "Low" in preset_choice: multiplier = 0.1
    elif "High" in preset_choice: multiplier = 1.8
    features_payload = {f: multiplier for f in expected_features}

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Live Inference", "📊 Feature Analytics", "📜 Execution History"])

    # --- TAB 1: Live Inference ---
    with tab1:
        left_col, right_col = st.columns([1.2, 1])
        with left_col:
            st.subheader("⚙️ Payload Configuration")
            with st.expander("🔍 View Full JSON Matrix", expanded=False):
                st.json(features_payload)
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🧠 Execute AI Prediction", type="primary", width='stretch')

        with right_col:
            st.subheader("🎯 Inference Output")
            if predict_btn:
                with st.spinner("Processing computations..."):
                    try:
                        response = requests.post(f"{api_url}/predict", json={"features": features_payload})
                        if response.status_code == 200:
                            tc_result = response.json().get("predicted_critical_temp_K")
                            
                            # Save to history
                            st.session_state.history.append({"Profile": preset_choice.split(" ")[0], "Predicted Tc (K)": round(tc_result, 2)})
                            
                            # Speedometer / Gauge Chart
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=tc_result,
                                title={'text': "Critical Temp (K)", 'font': {'color': 'white', 'size': 20}},
                                gauge={
                                    'axis': {'range': [0, 150], 'tickcolor': "white"},
                                    'bar': {'color': "#38bdf8"},
                                    'bgcolor': "rgba(0,0,0,0)",
                                    'steps': [
                                        {'range': [0, 40], 'color': "rgba(56, 189, 248, 0.2)"},
                                        {'range': [40, 90], 'color': "rgba(74, 222, 128, 0.3)"},
                                        {'range': [90, 150], 'color': "rgba(250, 204, 21, 0.4)"}
                                    ],
                                }
                            ))
                            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300, margin=dict(l=20, r=20, t=50, b=20))
                            st.plotly_chart(fig_gauge, width='stretch')
                            
                            if tc_result > 40:
                                st.balloons()
                                st.success("🔥 **High-Tc Material Detected!**")
                            else:
                                st.info("❄️ **Standard Material Range.**")
                        else:
                            st.error("Inference Failed.")
                    except Exception as ex:
                        st.error(f"API Error: {ex}")
            else:
                st.info("System Ready. Click Execute AI Prediction to begin.")

    # --- TAB 2: Feature Analytics ---
    with tab2:
        st.subheader("📈 Multi-Dimensional Feature Analysis")
        top_15_features = expected_features[:15]
        
        c1, c2 = st.columns(2)
        with c1:
            # Bar Chart
            chart_data = pd.DataFrame({"Feature Name": top_15_features, "Value": [features_payload[f] for f in top_15_features]}).set_index("Feature Name")
            st.bar_chart(chart_data)
            
        with c2:
            # Radar Chart
            df_radar = pd.DataFrame(dict(r=[features_payload[f] for f in top_15_features], theta=top_15_features))
            fig_radar = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
            fig_radar.update_traces(fill='toself', line_color='#4ade80')
            fig_radar.update_layout(paper_bgcolor="rgba(0,0,0,0)", polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, color="white")), font_color="white", height=350)
            st.plotly_chart(fig_radar, width='stretch')

    # --- TAB 3: Execution History ---
    with tab3:
        st.subheader("📜 Session Prediction Log")
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, width='stretch')
            
            # Download Button
            csv = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Report as CSV", data=csv, file_name='superconductor_predictions.csv', mime='text/csv')
        else:
            st.warning("No predictions run yet. Execute a prediction to see history here.")