
import streamlit as st  # type: ignore
import os
import sys
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.loader import load_deap_raw  # type: ignore

st.set_page_config(page_title="NeuroAI Dashboard", page_icon="🧠", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

#  Header Section
st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h1 style="font-size: 42px; margin-bottom: 5px;">🧠 NeuroAI – EEG-Based Emotion & Focus Detector</h1>
    <p style="color: #6C757D; font-size: 18px;">AI system that analyzes EEG signals to detect emotional state and focus level.</p>
</div>
<hr>
""", unsafe_allow_html=True)

# Layout: Left side Upload/Settings, Right side Signal Visualization
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("###  EEG Data Input")
    
    # Dataset selector
    dataset = st.selectbox("Select Dataset Format (Optional)", ["DEAP", "SEED", "Custom"])
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload EEG Data (.edf, .bdf, .csv)", type=['bdf', 'dat', 'mat', 'pkl', 'edf', 'csv'])
    
    if uploaded_file is not None:
        # Save logic
        temp_dir = "data/raw/"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"✅ File Received: {uploaded_file.name}")
        st.session_state['current_file_path'] = file_path
        st.session_state['current_dataset'] = dataset
        
        if st.button("Run Analysis & View Dashboard 🚀", use_container_width=True, type="primary"):
            st.switch_page("pages/2_Dashboard.py")

with col_right:
    st.markdown("###  EEG Signal Visualization")
    
    if uploaded_file is not None:
        with st.spinner("Extracting signal for visualization..."):
            try:
                # Basic attempt to read the file for visualization
                file_path = st.session_state.get('current_file_path', '')
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    # Use first 4 numeric columns
                    num_cols = df.select_dtypes(include=[np.number]).columns[:4]
                    if len(num_cols) > 0:
                        plot_data = {col: df[col].values for col in num_cols}
                    else:
                        plot_data = None
                else:
                    # Try using mne through load_deap_raw
                    raw = load_deap_raw(file_path)
                    if hasattr(raw, 'get_data'):
                        data = raw.get_data()
                        ch_names = raw.info['ch_names'][:4] # first 4 channels
                        plot_data = {ch: data[i, :1000] for i, ch in enumerate(ch_names)} # plot first 1000 points
                    elif isinstance(raw, dict):
                        keys = [k for k in raw.keys() if 'data' in k]
                        if keys:
                            data = raw[keys[0]]
                            if hasattr(data, 'ndim') and data.ndim == 3:
                                data = data.reshape(-1, data.shape[-1])
                            plot_data = {f"Ch{i+1}": data[i, :1000] for i in range(min(4, data.shape[0]))}
                        else:
                            plot_data = None
                    else:
                        data = raw
                        n_ch = data.shape[0] if hasattr(data, 'shape') else len(data)
                        plot_data = {f"Ch{i+1}": data[i, :1000] for i in range(min(4, n_ch))}

                if isinstance(plot_data, dict):
                    fig = go.Figure()
                    # Offset for vertical stacking
                    offset = 0
                    for ch_name, signal in plot_data.items():
                        # Normalize signal to have unit variance for plotting and add offset
                        if np.std(signal) > 0:
                            signal = (signal - np.mean(signal)) / np.std(signal)
                        fig.add_trace(go.Scatter(y=signal + offset, name=ch_name, mode='lines', line={"width": 1}))
                        offset -= 5 # Stack downwards
                    
                    fig.update_layout(
                        title="Raw EEG Signals (Channels over Time)",
                        xaxis_title="Time / Samples",
                        yaxis_title="Channels",
                        yaxis={"showticklabels": False}, # Hide y-axis numbers
                        height=400,
                        margin={"l": 0, "r": 0, "t": 40, "b": 0},
                        plot_bgcolor='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not extract signal data for visualization from this file format.")
                    
            except Exception as e:
                st.warning(f"Could not load real data for visualization: {e}. Showing sample data.")
                # Fallback synthetic plot
                t = np.linspace(0, 10, 1000)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=t, y=np.sin(2*np.pi*1*t) + np.random.randn(1000)*0.2, name="Fp1"))
                fig.add_trace(go.Scatter(x=t, y=np.sin(2*np.pi*1.5*t) + np.random.randn(1000)*0.2 - 3, name="Fp2"))
                fig.add_trace(go.Scatter(x=t, y=np.sin(2*np.pi*2*t) + np.random.randn(1000)*0.2 - 6, name="C3"))
                fig.update_layout(title="EEG Signal Plot (Sample Data)", height=400, yaxis={"showticklabels": False}, plot_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
                
    else:
        st.info("Upload an EEG data file to preview the signal waveforms.")
        
        # Show an empty placeholder chart
        fig = go.Figure()
        fig.update_layout(
            title="EEG Signal Plot",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            height=400,
            plot_bgcolor='#f8f9fa' # light grey
        )
        # Add some text in the middle
        fig.add_annotation(text="No data uploaded yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font={"size": 20, "color": "gray"})
        st.plotly_chart(fig, use_container_width=True)
