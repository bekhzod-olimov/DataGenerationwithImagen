# Import libraries
import torch, numpy as np
from imagen_pytorch import Unet, Imagen

# Set seed to ensure reproducibility
torch.manual_seed(2023); np.random.seed(2023)
    
def get_imagen(text_embed_dim, im_size):
    
    """
    
    This function gets several parameters and returns a model to be trained.
    
    Parameters:
    
        text_embed_dim     - dimension for text embedding, int;
        im_size            - input image size, int. 
    
    """
    
    unet = Unet(
        dim = 128,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 1,
        layer_attns = (False, False, False, True),
        layer_cross_attns = False
    )

    # Imagen model, which contains the UNet above

    imagen = Imagen(
        condition_on_text = True,  
        text_embed_dim = text_embed_dim,
        unets = unet,
        image_sizes = im_size,
        timesteps = 100
    )
    
    return imagen
