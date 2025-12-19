import torch
from torch import pi

def generate_smiley(n_samples:int = 2000,
                    variance_left_eye:float = 0.1,
                    variance_right_eye:float = 0.1,
                    position_left_eye:list = [-0.5, 0.5],
                    position_right_eye:list = [0.5, 0.5]
                   ) -> torch.tensor:
    """
    Create data resembling a smiley face. You can change parameters to move and 
    change size of each eye separately.

    Returns a torch.tensor of size (n_smaples, 2)
    """
    # Split the sample: 1/4 for each eye and the other half for the mouth
    n_points = n_samples // 4
    
    # Eyes (Gaussians)
    left_eye = torch.randn(n_points, 2) * variance_left_eye + torch.tensor(position_left_eye)
    right_eye = torch.randn(n_points, 2) * variance_right_eye + torch.tensor(position_right_eye)
    
    # Mouth
    n_mouth =  2 * n_points
    theta = torch.rand(n_mouth) * pi  
    r = 0.6 + torch.randn(n_mouth) * 0.05

    # Transform into cartesian coordinates
    mouth_x = r * torch.cos(theta)
    mouth_y = -r * torch.sin(theta) + 0.1
    mouth = torch.stack([mouth_x, mouth_y],dim=1)
    
    data = torch.cat([left_eye, right_eye, mouth], dim=0)
    
    # Shuffle data
    idx = torch.randperm(data.size(0))
    return data[idx]


def generate_moons(n_samples:int=2000) -> torch.tensor:
    """
    Create data of two moons (classic non linear separable dataset)

    Returns a torch.tensor of size (n_smaples, 2)
    """
    # Split the sample: half for each moon 
    n_moon = n_samples // 2

    # Upper moon
    theta = torch.rand(n_moon) * pi  
    r = 1 + torch.randn(n_moon) * 0.05

    # Transform into cartesian coordinates
    up_moon_x =  r * torch.cos(theta) + 0.5
    up_moon_y = r * torch.sin(theta) - 0.25
    upper_moon = torch.stack([up_moon_x, up_moon_y],dim=1)
    
    # Lower moon
    theta = torch.rand(n_moon) * pi  
    r = 1 + torch.randn(n_moon) * 0.05

    # Transform into cartesian coordinates
    low_moon_x =  r * torch.cos(theta) - 0.5
    low_moon_y = -r * torch.sin(theta) + 0.25
    lower_moon = torch.stack([low_moon_x, low_moon_y],dim=1)

    data = torch.cat([lower_moon, upper_moon], dim=0)
    
    # Shuffle data
    idx = torch.randperm(data.size(0))
    return data[idx]

    
### Define your own custom generation function and import it on the notebook!