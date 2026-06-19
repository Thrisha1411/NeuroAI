import streamlit as st
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

st.set_page_config(page_title="Admin", page_icon="🛡", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Header
st.markdown("""
<div style="background: #311B92; padding: 20px; border-radius: 0 0 16px 16px; margin: -60px -60px 30px -60px; color: white;">
    <div style="max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center; padding: 0 20px;">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 28px; margin-right: 15px;">🛡</div>
            <div>
                <h2 style="color: white; margin: 0;">Admin Panel</h2>
                <div style="opacity: 0.8; font-size: 14px;">System Health & Logs</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Status Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-label">Status</div><div class="stat-value" style="color: #2e7d32;">Online</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-label">Analyses</div><div class="stat-value">8</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-label">Logs</div><div class="stat-value">12</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-label">Version</div><div class="stat-value">v2.1</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Grid: Left (Distributions) - Right (Logs)
col_left, col_right = st.columns([1, 1])

with col_left:
    # Emotion Distribution
    st.markdown('<div class="card"><h5>Emotion Distribution</h5>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <div style="display:flex; justify-content:space-between; font-size: 11px;"><span>Angry</span><span>13%</span></div>
        <div style="width: 100%; height: 6px; background: #eee; border-radius: 3px;"><div style="width: 13%; height: 100%; background: #E91E63; border-radius: 3px;"></div></div>
    </div>
    <div style="margin-bottom: 8px;">
        <div style="display:flex; justify-content:space-between; font-size: 11px;"><span>Happy</span><span>25%</span></div>
        <div style="width: 100%; height: 6px; background: #eee; border-radius: 3px;"><div style="width: 25%; height: 100%; background: #9C27B0; border-radius: 3px;"></div></div>
    </div>
    <div style="margin-bottom: 8px;">
        <div style="display:flex; justify-content:space-between; font-size: 11px;"><span>Calm</span><span>25%</span></div>
        <div style="width: 100%; height: 6px; background: #eee; border-radius: 3px;"><div style="width: 25%; height: 100%; background: #2196F3; border-radius: 3px;"></div></div>
    </div>
    <div style="margin-bottom: 8px;">
        <div style="display:flex; justify-content:space-between; font-size: 11px;"><span>sad</span><span>25%</span></div>
        <div style="width: 100%; height: 6px; background: #eee; border-radius: 3px;"><div style="width: 25%; height: 100%; background: #000000; border-radius: 3px;"></div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Focus Distribution (Stacked below Emotion)
    st.markdown('<div class="card"><h5>Focus Distribution</h5>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <div style="font-size: 12px; font-weight: 500;">Focused (75%)</div>
        </div>
        <div style="width: 100%; height: 6px; background: #eee; border-radius: 3px;">
            <div style="width: 75%; height: 100%; background: linear-gradient(90deg, #9D4EDD, #FF006E); border-radius: 3px;"></div>
        </div>
    </div>
    <div style="margin-bottom: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <div style="font-size: 12px; font-weight: 500;">Distracted (25%)</div>
        </div>
        <div style="width: 100%; height: 6px; background: #eee; border-radius: 3px;">
            <div style="width: 25%; height: 100%; background: #78909C; border-radius: 3px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card" style="height: 100%; display: flex; flex-direction: column;"><h5>System Logs</h5>', unsafe_allow_html=True)
    
    # Mock Logs
    logs = [
        "[INFO] System initialized successfully.",
        f"[INFO] User session started at {datetime.now().strftime('%H:%M:%S')}",
        "[DATA] Loaded DEAP dataset (32 channels).",
        "[MODEL] Inference run on 'subject_05.edf' - 88% Confidence.",
        "[WARN] Artifact detection: High noise in channel Fp1.",
        "[INFO] Database connection active.",
        "[API] Request to /predict received.",
        "[INFO] Cleanup temp files..."
    ]

    # Create columns for Log View and Actions Sidebar
    log_col, act_col = st.columns([3, 1])
    
    with log_col:
        # Adjusted height to fill the space
        log_html = '<div style="flex-grow: 1; background: #f8f9fa; border: 1px solid #eee; border-radius: 6px; padding: 15px; overflow-y: auto; font-family: monospace; font-size: 12px; color: #333; height: 320px;">'
        for log in logs:
            color = "#d63031" if "[WARN]" in log else "#0984e3" if "[INFO]" in log else "#2d3436"
            log_html += f'<div style="margin-bottom: 6px;"><span style="color: {color}; font-weight: bold;">{log.split("]")[0]}]</span> {log.split("]")[1]}</div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

    with act_col:
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True) # Spacer
        if st.button("⚡ Retrain Model", use_container_width=True):
            with st.spinner("Retraining model..."):
                time.sleep(2)
            st.success("Retrained!")
            time.sleep(1)
            st.rerun()
            
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) # Spacer

        if st.button("🗑 Clear Logs", use_container_width=True):
            st.toast("Logs cleared.")

    st.markdown('</div>', unsafe_allow_html=True)
