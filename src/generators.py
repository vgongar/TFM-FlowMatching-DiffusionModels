import torch 
import torch.nn as nn
import numpy as np

@torch.no_grad() # Don't calculate gradients in this function
def generate_euler(model:torch.nn, n:int=2000, steps:int=100, device = "cpu") -> np.array:
    """
    Simulates n points using Euler's integration method.
    """
    model.eval() # We set the network in evaluation mode instead of training

    noise = torch.randn(n, 2).to(device)
    x_t = [noise]
    h = 1.0 / steps

    # Euler Method
    # We can feed the network all sampled points at once
    # So Euler Method will be calculated on every row (each row is a sample point).
    for i in range(steps):
        
        t_val = i / steps
        t = torch.full((n,), t_val, device=device) # (n,1)
        
        x = x_t[-1] # (n,2)

        # Evaluate the net to get the velocities
        v = model(x, t) 
        
        # Update positions via Euler integration and add it to the trajectory
        x_t.append(x + v * h) # Euler
        
    return np.array(x_t)