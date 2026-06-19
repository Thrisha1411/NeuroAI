import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

st.set_page_config(page_title="History", page_icon="📜", layout="wide")

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), '..', 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Initialize base mock data if empty
if 'analysis_history' not in st.session_state:
    st.session_state['analysis_history'] = [
        {"Date": "Jan 05, 2026 19:29", "File": "subject_05_trial_31.edf", "Dataset": "DEAP", "Emotion": "Angry", "Focus": "Focused", "Conf": 88},
        {"Date": "Jan 05, 2026 19:29", "File": "participant_15.dat", "Dataset": "SEED", "Emotion": "Neutral", "Focus": "Distracted", "Conf": 77},
        {"Date": "Jan 05, 2026 19:29", "File": "custom_rec_001.csv", "Dataset": "Custom", "Emotion": "Happy", "Focus": "Focused", "Conf": 92},
        {"Date": "Jan 05, 2026 19:29", "File": "subject_19_trial_04.edf", "Dataset": "DEAP", "Emotion": "Calm", "Focus": "Focused", "Conf": 86},
        {"Date": "Jan 05, 2026 19:29", "File": "subject_01_trial_15.edf", "Dataset": "DEAP", "Emotion": "Happy", "Focus": "Focused", "Conf": 87},
        {"Date": "Jan 05, 2026 19:29", "File": "subject_08_trial_22.edf", "Dataset": "DEAP", "Emotion": "Calm", "Focus": "Focused", "Conf": 92},
        {"Date": "Jan 05, 2026 19:29", "File": "participant_03_A.dat", "Dataset": "SEED", "Emotion": "Positive", "Focus": "Focused", "Conf": 85},
        {"Date": "Jan 05, 2026 19:29", "File": "subject_12_trial_08.edf", "Dataset": "DEAP", "Emotion": "Sad", "Focus": "Distracted", "Conf": 79},
    ]

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px;">
        <div style="background: #e91e63; color: white; width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px;">📜</div>
        <div>
            <h3 style="margin: 0; color: #333; font-size: 24px;">Analysis History</h3>
            <div style="color: #666; font-size: 14px;">View all your previous EEG analyses</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div style="text-align: right;">', unsafe_allow_html=True)
    c_btn1, c_btn2 = st.columns([1, 1])
    with c_btn1:
        if st.button("Refresh"):
            st.rerun()
    with c_btn2:
        if st.button("New Analysis"):
            st.switch_page("Upload.py")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

df = pd.DataFrame(st.session_state['analysis_history'])

# Stats
total_analyses = len(df)
happy_emotions = len(df[df['Emotion'].str.lower().isin(['happy', 'positive'])]) if not df.empty else 0
focused_states = len(df[df['Focus'].str.lower() == 'focused']) if not df.empty else 0
avg_conf = int(df['Conf'].mean()) if not df.empty else 0

# Stats Row
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Analyses</div>
        <div class="stat-value">{total_analyses}</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="stat-card yellow-accent">
        <div class="stat-label">Positive/Happy</div>
        <div class="stat-value" style="color: #F57F17;">{happy_emotions}</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="stat-card purple-accent">
        <div class="stat-label">Focused States</div>
        <div class="stat-value" style="color: #7B1FA2;">{focused_states}</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="stat-card blue-accent">
        <div class="stat-label">Avg Confidence</div>
        <div class="stat-value" style="color: #00838F;">{avg_conf}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Filters
f1, f2, f3 = st.columns([1, 1, 4])
with f1:
    emo_filter = st.selectbox("Filter Emotion", ["All Emotions"] + sorted(list(df['Emotion'].unique())), label_visibility="collapsed")
with f2:
    foc_filter = st.selectbox("Filter Focus", ["All Focus Levels"] + sorted(list(df['Focus'].unique())), label_visibility="collapsed")

# Apply filters
if emo_filter != "All Emotions":
    df = df[df['Emotion'] == emo_filter]
if foc_filter != "All Focus Levels":
    df = df[df['Focus'] == foc_filter]

with f3:
    st.markdown(f'<div style="text-align: right; color: #888; padding-top: 10px; font-size: 13px;">{len(df)} results</div>', unsafe_allow_html=True)


# Construct HTML Table (No indentation in the string to prevent code blocks)
table_rows = ""
for i, row in df.iterrows():
    border_bottom = "border-bottom: 1px solid #E1E4E8;" if i < len(df) - 1 else ""
    
    # Emotion badge
    if row['Emotion'] in ["Angry", "Sad", "Neutral"]:
        em_style = "background-color: #FFEBEE; color: #D32F2F; border: 1px solid #FFCDD2;"
    elif row['Emotion'] in ["Happy", "Positive", "Calm"]:
        em_style = "background-color: #FFF3E0; color: #EF6C00; border: 1px solid #FFE0B2;"
    else:
        em_style = "background-color: #E3F2FD; color: #1976D2; border: 1px solid #BBDEFB;"
        
    # Focus badge
    if row['Focus'] == "Focused":
        fo_style = "background-color: #F3E5F5; color: #7B1FA2; border: 1px solid #E1BEE7;"
    else:
        fo_style = "background-color: #ECEFF1; color: #546E7A; border: 1px solid #CFD8DC;"

    table_rows += f"""<tr style="{border_bottom} background-color: white;">
<td style="padding: 16px 24px; color: #24292E; font-size: 13px;">{row['Date']}</td>
<td style="padding: 16px 24px; color: #24292E; font-size: 13px;"><span style="color: #9D4EDD; font-size: 14px; margin-right: 8px;">●</span>{row['File']}</td>
<td style="padding: 16px 24px;"><span style="background-color: #F1F8FF; color: #0366D6; border: 1px solid #c8e1ff; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500;">{row['Dataset']}</span></td>
<td style="padding: 16px 24px;"><span style="{em_style} padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">{row['Emotion']}</span></td>
<td style="padding: 16px 24px;"><span style="{fo_style} padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">{row['Focus']}</span></td>
<td style="padding: 16px 24px;"><div style="display: flex; align-items: center; gap: 10px;"><div style="flex-grow: 1; height: 6px; background-color: #E1E4E8; border-radius: 3px; max-width: 60px;"><div style="width: {row['Conf']}%; height: 100%; background-color: #9D4EDD; border-radius: 3px;"></div></div><span style="color: #586069; font-size: 12px;">{row['Conf']}%</span></div></td>
<td style="padding: 16px 24px; text-align: right;"><button style="background: white; border: 1px solid #D1D5DA; color: #586069; border-radius: 4px; padding: 4px 10px; font-size: 12px; cursor: pointer;">👁 View</button></td>
</tr>"""

if not table_rows:
    table_rows = """<tr><td colspan="7" style="padding: 20px; text-align: center; color: #888;">No history records found matching your filters.</td></tr>"""

table_html = f"""
<div class="card" style="padding: 0; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.03); border: 1px solid #eee;">
    <table style="width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif;">
        <thead>
            <tr style="background-color: #FAFBFC; border-bottom: 1px solid #E1E4E8;">
                <th style="text-align: left; padding: 16px 24px; color: #586069; font-weight: 600; font-size: 12px;">Date & Time</th>
                <th style="text-align: left; padding: 16px 24px; color: #586069; font-weight: 600; font-size: 12px;">File Name</th>
                <th style="text-align: left; padding: 16px 24px; color: #586069; font-weight: 600; font-size: 12px;">Dataset</th>
                <th style="text-align: left; padding: 16px 24px; color: #586069; font-weight: 600; font-size: 12px;">Emotion</th>
                <th style="text-align: left; padding: 16px 24px; color: #586069; font-weight: 600; font-size: 12px;">Focus</th>
                <th style="text-align: left; padding: 16px 24px; color: #586069; font-weight: 600; font-size: 12px;">Confidence</th>
                <th style="text-align: right; padding: 16px 24px; color: #586069; font-weight: 600; font-size: 12px;">Actions</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
</div>
"""

st.markdown(table_html, unsafe_allow_html=True)
