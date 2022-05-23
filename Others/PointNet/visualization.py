import numpy as np
import plotly.offline as pyo
from plotly.offline import init_notebook_mode
import plotly.graph_objects as go
pyo.init_notebook_mode()

def visualize(points, label=None, name=None, seg_num=None, fname=None):
    x, y, z = np.array(points).T
    if label is not None:
        label = np.array(label).T
    layout = go.Layout(
        scene=dict(
            aspectmode='data'

        ))
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers',
                                       marker=dict(
                                           size=15,              
                                           color=label,          
                                           colorscale='rainbow',
                                           opacity=1.0,
                                       ))],
                    layout_title_text=f"[ShapeNet Dataset]   Label: {name},   Segmentation parts: {seg_num},   Total Points: {label.shape[0] if label is not None else None}",
                    layout=layout)
    fig.update_traces(marker=dict(size=1.0,
                                  line=dict(width=1.0,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.show()
    fig.write_html(f'{fname}.html')
