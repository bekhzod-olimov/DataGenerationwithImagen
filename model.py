from imagen_pytorch import Unet, Imagen
import torch, numpy as np

torch.manual_seed(2023)
np.random.seed(2023)
    
def get_imagen(text_embed_dim, im_size):
    
    unet = Unet(
        dim = 128,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 1,
        layer_attns = (False, False, False, True),
        layer_cross_attns = False
    )

    # imagen, which contains the unet above

    imagen = Imagen(
        condition_on_text = True,  
        text_embed_dim = text_embed_dim,
        unets = unet,
        image_sizes = im_size,
        timesteps = 100
    )
    
    return imagen

