import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧠 Optimizer Path Comparison on a 3D Loss Surface")

st.markdown(
    """
This interactive visualization shows how different optimizers (SGD, Momentum, RMSprop, Adam) 
navigate a **2D loss surface**. Try tweaking parameters to see how they behave!
"""
)


# -- Define a smoother loss function
def loss_fn(x, y):
    return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)


def grad_fn(x, y):
    dx = np.cos(x) * np.cos(y) + 0.2 * x
    dy = -np.sin(x) * np.sin(y) + 0.2 * y
    return dx, dy


# -- Create the 3D surface
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = loss_fn(X, Y)

# -- Sidebar controls
optimizer_choice = st.sidebar.selectbox(
    "🧪 Optimizer", ["SGD", "Momentum", "RMSprop", "Adam"]
)
lr = st.sidebar.slider("🔧 Learning Rate", 0.001, 0.2, 0.05, 0.001)
steps = st.sidebar.slider("🔁 Optimization Steps", 10, 200, 100, 10)
start_x = st.sidebar.slider("📍 Start X", -3.0, 3.0, -2.5, 0.1)
start_y = st.sidebar.slider("📍 Start Y", -3.0, 3.0, -2.5, 0.1)

# -- Optimization
path_x, path_y, path_z = [start_x], [start_y], [loss_fn(start_x, start_y)]

# Optimizer states
v = 0  # for momentum
m, v_adam = 0, 0  # for Adam
s = 0  # for RMSprop
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8

x_t, y_t = start_x, start_y

for t in range(1, steps + 1):
    dx, dy = grad_fn(x_t, y_t)

    if optimizer_choice == "SGD":
        x_t -= lr * dx
        y_t -= lr * dy

    elif optimizer_choice == "Momentum":
        v_x = beta1 * v + (1 - beta1) * dx
        v_y = beta1 * v + (1 - beta1) * dy
        x_t -= lr * v_x
        y_t -= lr * v_y
        v = v_x  # store last

    elif optimizer_choice == "RMSprop":
        s = beta2 * s + (1 - beta2) * (dx**2 + dy**2)
        x_t -= lr * dx / (np.sqrt(s) + epsilon)
        y_t -= lr * dy / (np.sqrt(s) + epsilon)

    elif optimizer_choice == "Adam":
        m = beta1 * m + (1 - beta1) * (dx + dy)
        v_adam = beta2 * v_adam + (1 - beta2) * ((dx + dy) ** 2)
        m_hat = m / (1 - beta1**t)
        v_hat = v_adam / (1 - beta2**t)
        x_t -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        y_t -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

    path_x.append(x_t)
    path_y.append(y_t)
    path_z.append(loss_fn(x_t, y_t))

# -- Create 3D Plotly figure
fig = go.Figure()

# Surface
fig.add_trace(
    go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", opacity=0.85, showscale=False)
)

# Optimizer Path
fig.add_trace(
    go.Scatter3d(
        x=path_x,
        y=path_y,
        z=path_z,
        mode="lines+markers",
        marker=dict(size=4, color="red"),
        line=dict(color="red", width=3),
        name="Optimizer Path",
    )
)

fig.update_layout(
    title=f"{optimizer_choice} Path on 3D Loss Surface",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="Loss",
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    height=700,
)

st.plotly_chart(fig, use_container_width=True)
