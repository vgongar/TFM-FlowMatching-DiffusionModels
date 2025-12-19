import torch
import torch.nn as nn

class VectorFieldNet(nn.Module): # Neural network for the vector field
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), # Input: 2 coordinates (x,y) + 1 time (t) = 3
            nn.SiLU(), 
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2) # Output: Velocity (vx, vy)
        )

    def forward(self, x: torch.tensor, t):
        # Concatenate inputs x and t into a single column vector
        t_vec = t.view(-1, 1)
        x_input = torch.cat([x, t_vec], dim=-1)
        return self.net(x_input)