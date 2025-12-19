import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm # Importar mapas de color

def create_flow_gif(generated_data, original_data = None, filename="flow_animation.gif", colormap:str = 'viridis'):
    """
    Assemble a gif in which each frame the color of the points change according to the progress
    of the animation

    Possible colormaps: 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm'
    """
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis('off')
    
    scat = ax.scatter([], [], s=10, alpha=0.7)

    if original_data is not None:
        background = ax.scatter(original_data[:,0], original_data[:,1], s=1, alpha=0.1, c='red')
    
    # Text to show time
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)

    colormap = plt.get_cmap(colormap) 
    
    total_frames = len(generated_data)

    # 3. Función de actualización
    def update(frame_idx):
        # Datos actuales
        current_data = generated_data[frame_idx]
        scat.set_offsets(current_data) # Update the scatterplot
        
        # Progress
        t_normalized = frame_idx / (total_frames - 1)
        
        # Set the color to the current time
        scat.set_color(colormap(t_normalized))
        
        # Update timestamp
        time_text.set_text(f"t = {t_normalized:.2f}")
        
        return scat, time_text

    # 4. Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=total_frames, interval=40, blit=True
    )

    # 5. Guardar
    anim.save(filename, writer='pillow', fps=25)
    plt.close(fig)
    
    return filename