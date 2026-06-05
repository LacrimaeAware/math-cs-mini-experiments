"""
region_plot_3d.py — interactive 3D plot of a solid region (calculus viz).

Plots the surfaces z = x - y and z = x + y over the triangle 0 <= y <= x <= 3,
i.e. the region { 0 <= x <= 3, 0 <= y <= x, x - y <= z <= x + y }, with edge and
ridge lines for clarity. Opens an interactive figure in the browser.

Requires plotly (not in pyproject deps): pip install plotly
"""

import numpy as np
import plotly.graph_objects as go

# Define the region
x = np.linspace(0, 3, 80)
y = np.linspace(0, 3, 80)
X, Y = np.meshgrid(x, y)
mask = (Y <= X)

Z1 = np.where(mask, X - Y, np.nan)
Z2 = np.where(mask, X + Y, np.nan)

# Create interactive 3D figure
fig = go.Figure()

# Lower and upper surfaces
fig.add_surface(x=X, y=Y, z=Z1, colorscale="Blues", opacity=0.6, name="z = x - y")
fig.add_surface(x=X, y=Y, z=Z2, colorscale="Oranges", opacity=0.6, name="z = x + y")

# Add edges for clarity
for x0 in [0, 3]:
    y_edge = np.linspace(0, x0, 50)
    fig.add_scatter3d(x=[x0]*len(y_edge), y=y_edge, z=x0 - y_edge, mode='lines', line=dict(color='blue'))
    fig.add_scatter3d(x=[x0]*len(y_edge), y=y_edge, z=x0 + y_edge, mode='lines', line=dict(color='orange'))

# Ridge line (y=0, z=x)
x_ridge = np.linspace(0, 3, 50)
fig.add_scatter3d(x=x_ridge, y=np.zeros_like(x_ridge), z=x_ridge, mode='lines', line=dict(color='black'))

fig.update_layout(
    title="Region E: 0 ≤ x ≤ 3, 0 ≤ y ≤ x, x−y ≤ z ≤ x+y",
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='z'
    ),
    showlegend=False
)

fig.show()
