import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

import base64 # For showing the animation

import sys
import os
sys.path.append(os.path.abspath('.'))

# Import all code in other folders
try:
    from src.model import VectorFieldNet
    from src.data_generation import generate_smiley, generate_moons 
    from src.generators import generate_euler
    from src.animation_gif import create_flow_gif
    
except ImportError:
    st.error("Import Error: Could be (not found src/ folder, not found model or data_generation modules or didn't find imports inside")
    st.stop()

# --- Configuration ---
st.set_page_config(page_title="Flow Matching Visualized", layout="wide")
DEVICE = "cpu" 

mse_loss_fn = torch.nn.MSELoss(reduction='mean')
def train_step(model, batch, optimizer, device="cpu", loss_function=mse_loss_fn):
    model.train()
    
    z = batch.to(device).float()
    t = torch.rand(z.shape[0], device=device).view(-1, 1)
    eps = torch.randn_like(z).to(device)
    
    x = t * z + (1 - t) * eps
    
    output = model(x, t)
    target = z - eps
    loss = loss_function(output, target)
    
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def reset_animation() -> None:
    """
    Delete the gif from the state
    """
    st.session_state['gif_path'] = None 

def reset_all() -> None:
    """
    Delete all the state
    """
    st.session_state['training_data'] = None
    st.session_state['gif_path'] = None 
    
    
# --- State Handling ---
# We store variables so we don't forget (Streamlit reruns everytime a button is pressed for example)
if 'data_type' not in st.session_state:
    st.session_state['data_type'] = "Smiley" # Smiley by default
if 'training_data' not in st.session_state:
    st.session_state['training_data'] = None
if 'gif_path' not in st.session_state:
    st.session_state['gif_path'] = None

    
# --- Interface ---
st.title("🌬️ Visualizing Flow Matching")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Get your data")
    st.session_state['data_type'] = st.radio("Choose between the smiley or moons presets or draw your own:", ["Smiley", "Moons", "Draw"], on_change=reset_all)
    
    if st.session_state['data_type'] == "Smiley":
        if st.button("Generate"):
            reset_animation()
            st.session_state['training_data'] = generate_smiley(2000)
            
    elif st.session_state['data_type'] == "Moons":
        if st.button("Generate"):
            reset_animation()
            st.session_state['training_data'] = generate_moons(2000)
    
            
with col2:
    if st.session_state['data_type'] == "Smiley" or st.session_state['data_type'] == "Moons":
        st.subheader("Learning configuration")
        epochs = st.slider("Epochs", 100, 500, 1000)
        lr = st.select_slider("Learning Rate", options=[1e-2, 1e-3, 1e-4], value=1e-3)
        start_btn = st.button("🔥 Launch training", type="primary", disabled=(st.session_state['training_data'] is None))
    else: # Draw mode
        st.write("Draw in the square below:")
            
        canvas_result = st_canvas(
            stroke_width=10,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=300, width=300,
            drawing_mode="freedraw",
            key="canvas",
        )
            
        if canvas_result.image_data is not None:
            # Detect white pixels
            indices = np.where(canvas_result.image_data[:, :, 0] > 0)
            if len(indices[0]) > 0:
                # Normalize coordinates and center data
                x = (indices[1] - 150) / 60.0
                y = -(indices[0] - 150) / 60.0 
                
                reset_animation()
                st.session_state['training_data'] = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32)
                st.caption(f"Points captured: {len(st.session_state['training_data'])}")
            else:
                st.warning("Draw something!")


# Below the columns
if st.session_state['training_data'] is not None and (st.session_state['data_type'] == "Smiley" or st.session_state['data_type'] == "Moons"):
    data = st.session_state['training_data']
    with st.container(horizontal=True, horizontal_alignment="center"):
        st.space('stretch')
        st.scatter_chart(data={"x" : data[:,0], "y" : data[:,1]}, x="x", y="y", width="content", x_label="", y_label="")
        st.space('stretch')
elif st.session_state['data_type'] == "Draw":
    st.subheader("Learning configuration")
    epochs = st.slider("Epochs", 100, 500, 1000)
    lr = st.select_slider("Learning Rate", options=[1e-2, 1e-3, 1e-4], value=1e-3)
    
    start_btn = st.button("🔥 Launch training", type="primary", disabled=(st.session_state['training_data'] is None))
        
data = st.session_state['training_data'] # Retrieve data in case of forget

            
st.divider()


st.subheader("Animation visualizer")
col3, _ , col4 = st.columns([1, 0.5,1])
with col3:
    loss_chart = st.empty()       # Space reserved for the loss evolution

if start_btn and st.session_state['training_data'] is not None:
    # Instantiate model
    model = VectorFieldNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # For efficient batch handling
    #dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True)
    
    progress_bar = st.progress(0)
    loss_history = []
    
    # Training in epochs
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            loss = train_step(model, batch, optimizer, DEVICE)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        # Update progress bar (epochs start at 0)
        progress_bar.progress((epoch + 1) / epochs)
        
        # Loss and model visualization
        if epoch % 20 == 0 or epoch == epochs - 1:
            # Mostrar Loss
            loss_chart.line_chart(loss_history)
    with col3:        
        st.success("Training ended!")
    with col4:
        with st.spinner("Calculating trajectories and animating..."):
            
            generated_data = generate_euler(model, n=2000, steps=100)
            
            gif_path = create_flow_gif(generated_data, data)
            
            st.session_state['gif_path'] = gif_path
        

# Visualization
if st.session_state['gif_path'] is not None:
    with col4:
        try:
            file_ = open(f"{st.session_state['gif_path']}", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()
            
            st.image(
                st.session_state['gif_path'],
                caption="Simultaneous generation (From noise to data)",
                width="stretch"
            )
            
            with open(st.session_state['gif_path'], "rb") as file:
                st.download_button(
                    label="💾 Download animation",
                    data=file,
                    file_name="animation.gif",
                    mime="image/gif"
                )
        except FileNotFoundError:
            # Si el archivo se borró manualmente pero el estado sigue apuntando a él
            st.warning("No animation, please train a model")