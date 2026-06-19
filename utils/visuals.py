import plotly.graph_objects as go
import numpy as np
import plotly.express as px

def create_3d_brain(emotion=None):
    """
    Creates a 3D scatter plot representing a brain.
    Color changes based on emotion.
    """
    # Generate dummy brain points (simulated as a cloud)
    # In a real app, use a mesh or specific electrode coordinates
    
    # Sphere approximate
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    phi, theta = np.meshgrid(phi, theta)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Flatten
    x = x.flatten()
    y = y.flatten() * 1.2 # Elongate slightly
    z = z.flatten() * 0.8
    
    # Color map
    colors = {
        'Happy': '#FFD700', # Gold
        'Calm': '#00F3FF', # Cyan
        'Sad': '#BC13FE', # Purple
        'Angry': '#FF003C', # Red
        None: '#555555' # Grey
    }
    
    color = colors.get(emotion, '#555555')
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker={
            "size": 4,
            "color": color,
            "opacity": 0.6,
            "line": {"width": 0}
        }
    )])
    
    # Layout
    fig.update_layout(
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "bgcolor": 'rgba(0,0,0,0)'
        },
        paper_bgcolor='rgba(0,0,0,0)',
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        showlegend=False
    )
    
    return fig

def create_eeg_plot(data, fs=128):
    """
    Detailed EEG plot.
    """
    # Plot only first few channels for clarity
    n_channels_to_plot = 4
    samples = 500 # Show last ~4s
    
    if data.shape[1] > samples:
        plot_data = data[:n_channels_to_plot, -samples:]
    else:
        plot_data = data[:n_channels_to_plot, :]
        
    time = np.arange(plot_data.shape[1]) / fs
    
    fig = go.Figure()
    
    offsets = np.arange(n_channels_to_plot) * 3 # Separation
    
    colors = ['#00f3ff', '#bc13fe', '#0aff00', '#ff003c']
    
    for i in range(n_channels_to_plot):
        fig.add_trace(go.Scatter(
            x=time, 
            y=plot_data[i] + offsets[i],
            mode='lines',
            line={"color": colors[i % len(colors)], "width": 1.5},
            name=f'Ch {i+1}'
        ))
        
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={"color": '#e0e0e0', "family": 'Roboto Mono'},
        xaxis={"showgrid": False, "title": 'Time (s)'},
        yaxis={"showgrid": False, "showticklabels": False},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1}
    )
    
    return fig
