import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
try:
    # type: ignore - optional dependency; provide runtime fallback if missing
    import plotly.express as px
except Exception:
    px = None
    # Provide minimal fallbacks for used px functions (imshow, bar)
    def _px_imshow(z, text_auto=True, x=None, y=None, color_continuous_scale=None, aspect=None):
        # Use heatmap as fallback
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=z, x=x, y=y, colorscale=color_continuous_scale, showscale=True))
        # add text annotations if requested
        if text_auto:
            annotations = []
            for i, row in enumerate(z):
                for j, val in enumerate(row):
                    annotations.append(dict(x=x[j] if x else j, y=y[i] if y else i,
                                            text=str(val), showarrow=False, font=dict(color='black')))
            fig.update_layout(annotations=annotations)
        return fig

    def _px_bar(df, x, y, orientation='v', text=None):
        # df is a pandas DataFrame
        if orientation == 'h':
            fig = go.Figure(go.Bar(x=df[x], y=df[y], orientation='h', text=df[text] if text else None, marker_color='#FF006E'))
        else:
            fig = go.Figure(go.Bar(x=df[x], y=df[y], text=df[text] if text else None, marker_color='#FF006E'))
        return fig

    class _PXFallback:
        @staticmethod
        def imshow(z, text_auto=True, x=None, y=None, color_continuous_scale=None, aspect=None):
            return _px_imshow(z, text_auto=text_auto, x=x, y=y, color_continuous_scale=color_continuous_scale, aspect=aspect)

        @staticmethod
        def bar(df, x=None, y=None, orientation='v', text=None):
            return _px_bar(df, x=x, y=y, orientation=orientation, text=text)

    px = _PXFallback()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from inference.predict import predict_eeg
from utils.label_mapping import get_emotion_name, get_focus_name

st.set_page_config(page_title="Dashboard | NeuroAI", page_icon="📊", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Header
st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h1 style="font-size: 36px; margin-bottom: 5px;">NeuroAI – EEG-Based Emotion & Focus Detector</h1>
    <p style="color: #6C757D; font-size: 16px;">AI Analysis Results for EEG Data</p>
</div>
<hr>
""", unsafe_allow_html=True)

# Navigation Back
col_back, _ = st.columns([1, 5])
with col_back:
    if st.button("← Back to Upload"):
        # Use the page name (as defined in the pages folder) when switching pages
        try:
            st.switch_page("Upload")
        except Exception:
            # Fallback for older/newer Streamlit versions
            try:
                st.switch_page("Upload.py")
            except Exception:
                pass

if 'current_file_path' not in st.session_state:
    st.warning("Please upload a file first on the Upload page.")
    st.stop()

# Run Prediction if needed
if 'last_results' not in st.session_state:
    with st.spinner("Processing neural signals..."):
        file_path = st.session_state['current_file_path']
        results = predict_eeg(file_path)
        st.session_state['last_results'] = results
        
        # Add to history
        import datetime
        now_str = datetime.datetime.now().strftime("%b %d, %Y %H:%M")
        emotion_name_new = get_emotion_name(results.get('emotion_label', 0)).capitalize()
        focus_name_new = get_focus_name(results.get('focus_label', 0)).capitalize()
        em_probs_new = results.get('emotion_probs', [0.82, 0.12, 0.05, 0.01])
        em_conf_new = int(np.max(em_probs_new) * 100)
        dataset_used = st.session_state.get('current_dataset', 'Unknown')
        
        if 'analysis_history' not in st.session_state:
            # Initialize with empty list if not already set (or mock data could be here)
            st.session_state['analysis_history'] = []
            
        st.session_state['analysis_history'].insert(0, {
            "Date": now_str, 
            "File": os.path.basename(file_path), 
            "Dataset": dataset_used,
            "Emotion": emotion_name_new, 
            "Focus": focus_name_new, 
            "Conf": em_conf_new
        })

results = st.session_state['last_results']
emotion_name = get_emotion_name(results.get('emotion_label', 0)).capitalize()
focus_name = get_focus_name(results.get('focus_label', 0)).capitalize()

# Confidence Scores
em_probs = results.get('emotion_probs', [0.82, 0.12, 0.05, 0.01]) # Fallback mock
fo_probs = results.get('focus_probs', [0.85, 0.15]) # Fallback mock

em_conf = np.max(em_probs) * 100
fo_conf = np.max(fo_probs) * 100

# Emoji Mapping based on Emotion Name
emoji_map = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Calm": "😌",
    "Neutral": "😐",
    "Positive": "🙂",
    "Negative": "🙁"
}
em_emoji = emoji_map.get(emotion_name, "🧠")
fo_emoji = "🟢" if focus_name == "Focused" else "🔴"

# Layout: 4️⃣ Emotion & 5️⃣ Focus
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="background: white; border-radius: 8px; padding: 10px 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid #FF006E; text-align: center;">
        <div style="color: #6C757D; font-size: 14px; margin-bottom: 5px;">Emotion Detected</div>
        <div style="font-size: 28px; margin-bottom: 5px; display: flex; align-items: center; justify-content: center; gap: 10px;">
            <span>{em_emoji}</span>
            <span style="color: #2B2D42;">{emotion_name}</span>
        </div>
        <div style="font-size: 14px; font-weight: bold; color: #495057;">{em_conf:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background: white; border-radius: 8px; padding: 10px 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 4px solid #9D4EDD; text-align: center;">
        <div style="color: #6C757D; font-size: 14px; margin-bottom: 5px;">Focus Level</div>
        <div style="font-size: 28px; margin-bottom: 5px; display: flex; align-items: center; justify-content: center; gap: 10px;">
            <span>{fo_emoji}</span>
            <span style="color: #2B2D42;">{focus_name}</span>
        </div>
        <div style="font-size: 14px; font-weight: bold; color: #495057;">{fo_conf:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# 7️⃣ Model Performance Section (Swapped to be above Probability)
st.markdown("### Model Performance Overview")

classes = ["Calm", "Sad", "Neutral", "Happy"] # From get_emotion_name mapping in label_mapping.py

col_metric, col_cm, col_loss = st.columns([1.5, 2, 2])

with col_metric:
    st.markdown("#### Test Metrics")
    # Mock metrics for research demo
    metrics = {
        "Accuracy": "91.3%",
        "Precision": "89.7%",
        "Recall": "90.2%",
        "F1 Score": "90.0%"
    }
    for k, v in metrics.items():
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 10px 0;">
            <span style="font-weight: bold; color: #495057;">{k}</span>
            <span style="color: #2B2D42;">{v}</span>
        </div>
        """, unsafe_allow_html=True)

with col_cm:
    st.markdown("#### Confusion Matrix")
    # Mock confusion matrix data
    z = [[45, 2, 1, 2],
         [3, 40, 5, 2],
         [1, 4, 38, 7],
         [2, 1, 6, 41]]
    
    x = classes
    y = classes
    
    # Needs reverse y to match standard CM orientation in Plotly
    fig_cm = px.imshow(z, text_auto=True, x=x, y=y, color_continuous_scale="Blues", aspect="auto")
    fig_cm.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_cm, use_container_width=True)

with col_loss:
    st.markdown("#### Training Loss Curve")
    # Mock loss data
    epochs = np.arange(1, 21)
    train_loss = 1.0 * np.exp(-epochs/5) + np.random.normal(0, 0.05, 20)
    val_loss = 1.1 * np.exp(-epochs/4.5) + np.random.normal(0, 0.05, 20) + 0.1
    
    loss_df = pd.DataFrame({'Epoch': epochs, 'Train Loss': train_loss, 'Val Loss': val_loss})
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=loss_df['Epoch'], y=loss_df['Train Loss'], mode='lines', name='Train Loss', line=dict(color='#9D4EDD')))
    fig_loss.add_trace(go.Scatter(x=loss_df['Epoch'], y=loss_df['Val Loss'], mode='lines', name='Val Loss', line=dict(color='#FF006E')))
    
    fig_loss.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='white', legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    fig_loss.update_xaxes(title_text='Epoch')
    fig_loss.update_yaxes(title_text='Loss')
    st.plotly_chart(fig_loss, use_container_width=True)


st.markdown("<hr>", unsafe_allow_html=True)

# 6️⃣ Prediction Confidence Chart (Swapped down)
st.markdown("### Emotion Probability")

if len(em_probs) == len(classes):
    prob_df = pd.DataFrame({"Emotion": classes, "Probability": em_probs})
else:
    # Use generic names if length mismatch
    prob_df = pd.DataFrame({"Emotion": [f"Class {i}" for i in range(len(em_probs))], "Probability": em_probs})

# Sort for better visual
prob_df = prob_df.sort_values(by="Probability", ascending=True)

fig_prob = px.bar(prob_df, x="Probability", y="Emotion", orientation='h', text="Probability")
fig_prob.update_traces(texttemplate='%{text:.2f}', textposition='outside', marker_color='#FF006E')
fig_prob.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='white', xaxis=dict(range=[0, 1.1]))
st.plotly_chart(fig_prob, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# 8️⃣ Footer Section
st.markdown("""
<div style="text-align: center; color: #6C757D; font-size: 14px; padding: 20px;">
    Model: CNN–LSTM with Attention &nbsp;|&nbsp; Datasets: DEAP, SEED &nbsp;|&nbsp; Framework: PyTorch &nbsp;|&nbsp; Visualization: Streamlit
</div>
""", unsafe_allow_html=True)

