import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# --- Page Config ---
st.set_page_config(
    page_title="VIBRATION COMMANDER AI",
    page_icon="�",
    layout="wide",
)

# --- Premium Cyber-Intelligence Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@300;400;700&display=swap');

    :root {
        --neon-blue: #00f2ff;
        --neon-pink: #ff007a;
        --bg-dark: #050505;
        --panel-bg: rgba(20, 20, 35, 0.7);
    }

    body, .stApp {
        background-color: var(--bg-dark);
        color: #e0e0e0;
        font-family: 'Outfit', sans-serif;
    }

    /* Sidebar - Input Zone */
    [data-testid="stSidebar"] {
        background-color: #0e0e1a;
        border-right: 1px solid var(--neon-blue);
    }
    
    /* Panel Glassmorphism */
    .glass-panel {
        background: var(--panel-bg);
        border: 1px solid rgba(0, 242, 255, 0.2);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Target Result Display */
    .intel-header {
        font-family: 'JetBrains Mono', monospace;
        color: var(--neon-blue);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }

    .prediction-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 0 10px var(--neon-blue);
    }

    /* Decision Engine Colors */
    .status-safe { color: #00ff88; border-left: 4px solid #00ff88; }
    .status-warning { color: #ffcc00; border-left: 4px solid #ffcc00; }
    .status-danger { color: var(--neon-pink); border-left: 4px solid var(--neon-pink); }

    .ai-box {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 15px;
        margin-top: 10px;
        font-size: 0.95rem;
    }

    /* Sidebar Input Styling */
    .stNumberInput label {
        color: #00f2ff !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Assets ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('Vibrations Detection\models_New\model.h5')
        scaler = joblib.load('Vibrations Detection\models_New\scaler.h5')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- Sidebar: Operational Control Center ---
with st.sidebar:
    st.markdown("## 🕹️ MISSION CONTROL")
    st.caption("Adjust system parameters below")
    st.divider()

    with st.expander("🌍 Drilling Parameters", expanded=True):
        dept = st.number_input('Measured Depth (m)', value=1300.0, step=10.0)
        wob = st.number_input('Weight on Bit (KLB)', value=20.0, step=0.1)
        rpm = st.number_input('Rotational Speed (RPM)', value=120.0, step=1.0)
        flow_in = st.number_input('Flow Rate (GPM)', value=800.0, step=10.0)
        torque = st.number_input('Torque (klb-ft)', value=15.0, step=0.5)
        spp = st.number_input('Circulation Press (PSI)', value=2500.0, step=10.0)
        rop = st.number_input('Rate of Penetration (m/hr)', value=10.0, step=0.1)
        
    with st.expander("💧 Mud Properties", expanded=True):
        mwt_in = st.number_input('Inlet Mud Weight (PPG)', value=12.0, step=0.1)
        mtemp_in = st.number_input('Inlet Temp (°C)', value=45.0, step=0.5)
        mtemp_out = st.number_input('Outlet Temp (°C)', value=55.0, step=0.5)

    st.write("")
    run_btn = st.button('🧠 EXECUTE AI ANALYSIS', use_container_width=True)

# --- Main Dashboard ---
col_head, col_logo = st.columns([4, 1])
with col_head:
    st.markdown("<h1 style='margin-bottom:0; color:#00f2ff;'>INTELLIGENT VIBRATION ANALYSIS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#777;'>Neural Network Engine for Drilling Optimization</p>", unsafe_allow_html=True)
with col_logo:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=80)

if model is None:
    st.error("Engine Offline: Model files missing.")
    st.stop()

# --- Placeholder / Results Area ---
if not run_btn:
    st.markdown("""
    <div class='glass-panel' style='text-align:center; padding:100px 20px;'>
        <h2 style='color:rgba(255,255,255,0.3);'>SYSTEM READY</h2>
        <p style='color:rgba(255,255,255,0.2);'>Waiting for Operational Input from Control Center (Sidebar)</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Prepare Data
    # Inputs = ['Depth', 'WOB', 'RPM', 'Q', 'T', 'SPP', 'ROP', 'M. Wt', 'M. T. in', 'M. T. out']
    input_data = pd.DataFrame([{
        'Depth': dept, 'WOB': wob, 'RPM': rpm, 'Q': flow_in,
        'T': torque, 'SPP': spp, 'ROP': rop, 'M. Wt': mwt_in,
        'M. T. in': mtemp_in, 'M. T. out': mtemp_out
    }])

    # Intelligence Simulation
    with st.spinner('Neural paths intersecting...'):
        time.sleep(0.5)
        input_scaled = scaler.transform(input_data)
        preds = model.predict(input_scaled)[0]

    # --- Result Infrastructure ---
    st.markdown("### 📡 REAL-TIME PREDICTIONS")
    
    res1, res2, res3 = st.columns(3)
    
    metrics = [
        {"name": "STICK-SLIP SEVERITY (SSS_H)", "val": preds[0], "thresh": 8.0, "unit": "Index"},
        {"name": "LATERAL VIBRATION (VIBXYH)", "val": preds[1], "thresh": 12.0, "unit": "g-RMS"},
        {"name": "AXIAL SHOCK (VIBZH)", "val": preds[2], "thresh": 6.0, "unit": "g-Peak"}
    ]

    for i, m in enumerate(metrics):
        with [res1, res2, res3][i]:
            status_class = "status-safe" if m['val'] < m['thresh'] else ("status-warning" if m['val'] < m['thresh']*1.5 else "status-danger")
            st.markdown(f"""
            <div class='glass-panel {status_class}'>
                <div class='intel-header'>{m['name']}</div>
                <div class='prediction-value'>{m['val']:.2f} <span style='font-size:1rem; color:#666;'>{m['unit']}</span></div>
            </div>
            """, unsafe_allow_html=True)

    # --- Intelligence Engine (AI Recommendations) ---
    st.divider()
    st.markdown("### 🤖 COGNITIVE DECISION ENGINE")
    
    advice_col, risk_col = st.columns([2, 1])

    with advice_col:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        st.write("**Drilling Optimization Strategy:**")
        
        has_issue = False
        # Stick-Slip Logic
        if preds[0] > 80.0:
            st.warning("⚠️ **Stick-Slip Detected:** Prediction indicates non-uniform rotation. **ACTION:** Increase RPM by 15% or decrease WOB to break the torque-cycle.")
            has_issue = True
        
        # Lateral Logic
        if preds[1] > 3.0:
            st.error("🔥 **Lateral Whirl Alert:** High energy lateral vibration. **ACTION:** Critical RPM range approaching. Drop RPM by 20 units and check for stabilizing flow rate.")
            has_issue = True

        # Axial Logic
        if preds[2] > 1.0:
            st.info("⚡ **Bit Bounce Noted:** Vertical acceleration is peaks. **ACTION:** Reduce WOB slightly or increase pump pressure to stiffen the string.")
            has_issue = True

        if not has_issue:
            st.success("✨ **Ideal Drilling Window:** Current parameters are perfectly balanced. System stability predicted. Maintain current parameters for maximum ROP.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    with risk_col:
        st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
        avg_risk = (preds[0]/8.0 + preds[1]/12.0 + preds[2]/6.0) / 3.0
        risk_text = "LOW" if avg_risk < 0.8 else ("MODERATE" if avg_risk < 1.2 else "CRITICAL")
        risk_color = "#00ff88" if risk_text == "LOW" else ("#ffcc00" if risk_text == "MODERATE" else "#ff007a")
        
        st.write("**Overall Integrity Risk:**")
        st.markdown(f"<h2 style='color:{risk_color}; text-align:center;'>{risk_text}</h2>", unsafe_allow_html=True)
        st.progress(min(avg_risk/2.0, 1.0))
        st.caption("Aggregated risk score based on tool-face stability and shock endurance.")
        st.markdown("</div>", unsafe_allow_html=True)


st.sidebar.markdown("---")
st.sidebar.caption("System: ANTIGRAVITY ENGINE V2.0")
st.sidebar.caption("Last Sync: " + time.strftime("%H:%M:%S"))
