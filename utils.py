# Import libraries
import torch, numpy as np, PIL.Image as Image
import torchvision.transforms.functional as F
from typing import List
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from einops import rearrange

# Set default options
DEFAULT_T5_NAME, T5_CONFIGS, MAX_LENGTH, = "google/t5-v1_1-base", {}, 256

# Function to convert tensor to numpy array
def tn_to_np(im): return (im * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def save_ims(gen_ims, save_path):
    
    """
    
    This function gets several parameters and saves images to the given path.
    
    Parameters:
    
        gen_ims    - generated images, list -> tensor;
        save_path  - a path to directory to save images.
        
    """
    
    # Go through generated images list
    for idx, gen_im in enumerate(gen_ims):
        
        # Convert tensor to numpy array
        im = tn_to_np(gen_im[0])
        # Save the image
        Image.fromarray(im).save(f"{save_path}/gen_im_{idx}.jpg")

# Function to check existence of a file
def exists(val): return val is not None

class T5LayerNorm(torch.nn.Module):
    
    """
        
    This class constructs a layernorm module in the T5 style without bias and no subtraction of mean.
    
    Parameters:
    
        hidden_size      - number of features in a hidden layer, int;
        eps              - epsilon value for variance, float.
        
    """
    
    def __init__(self, hidden_size, eps = 1e-6):
        
        super().__init__()
        self.weight, self.variance_epsilon = torch.nn.Parameter(torch.ones(hidden_size)), eps

    def forward(self, hidden_states):

        """
        
        This function gets hidden states and passes them through T5LayerNorm Class.
        
        Parameter:
        
            hidden_states    - hidden state volumes, tensor.
            
        Output:
        
            out              - output from the class after variance is applied, tensor.
        
        """

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # Half-precision conversion
        if self.weight.dtype in [torch.float16, torch.bfloat16]: hidden_states = hidden_states.to(self.weight.dtype)
        
        # Multiply weights with hidden states
        out = self.weight * hidden_states
        
        return out

class SquarePad:
  
  """
  
  This class gets an image and adds padding to make the image square.
  
  Parameter:
  
       image - an input image, array;
       
  Output:
  
       image - a square padded output image, array;
  
  """

  def __call__(self, image):
    
      # Get width and height of the image
      w, h = image.size
      
      # Get max values of the width and height
      max_wh = np.max([w, h])
      
      # Create padding
      hp = int((max_wh - w)/2)
      hp_rem = (max_wh - w)%2
      vp = int((max_wh - h)/2)
      vp_rem = (max_wh - h)%2
      padding = (hp, vp, hp + hp_rem, vp + vp_rem)
      
      # Apply padding and return the padded image
      return FF.pad(image, padding, 255, 'constant')

def insert_zeros(x, all_j):
    
    """
    
    This function gets input tensor and insert zeros to the pre-defined dimensions.
    
    Parameters:
    
        x       -  input tensor volume, tensor;
        all_j   -  number of dimensions to insert zeros, int.
        
    Output:
    
        out    - output tensor volume with zeros inserted, tensor.
    
    """
    
    # Create a tensor with zeros
    zeros_ = torch.zeros_like(x[:1])
    pieces = []
    i      = 0
    
    # Insert zeros
    for j in all_j + [len(x)]:
        pieces.extend([x[i:j], zeros_])
        i = j

    # Concatenate and return
    return torch.cat(pieces[:-1], dim = 0)

def default(val, d):
    if exists(val): return val
    return d() if callable(d) else d

def get_model_and_tokenizer(name):
    
    """
    
    This function gets a name parameter and returns a model and a tokenizer.
    
    Parameter:
    
        name      - name of the model and tokenizer, str;
        
    Output:
    
        model     - a model to be trained;
        tokenizer - tokenizer to tokenize inputs.
    
    """
    
    global T5_CONFIGS

    if name not in T5_CONFIGS: T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]: T5_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in T5_CONFIGS[name]: T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_model(name):
    
    """
    
    This function gets a name parameter and returns a model.
    
    Parameter:
    
        name      - name of the model and tokenizer, str;
        
    Output:
    
        model     - a model to be trained;
    
    """
    
    return T5EncoderModel.from_pretrained(name)


def t5_tokenize(texts: List[str], name = DEFAULT_T5_NAME):
    
    # Get model and tokenizer
    t5, tokenizer = get_model_and_tokenizer(name)

    # Switch to gpu
    if torch.cuda.is_available(): t5 = t5.cuda()
    
    # Get gpu name
    device = next(t5.parameters()).device

    # Get encoded inputs
    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    # Return input ids and attention mask
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)

def get_tokenizer(name):
    tokenizer = T5Tokenizer.from_pretrained(name, model_max_length=MAX_LENGTH)
    return tokenizer

def t5_encode_tokenized_text(
    token_ids, device, text_embed_dim,
    attn_mask = None,
    pad_id = None,
    name = DEFAULT_T5_NAME
):
    assert exists(attn_mask) or exists(pad_id)
    t5, _ = get_model_and_tokenizer(name)
    t5.encoder.final_layer_norm = torch.nn.Sequential(T5LayerNorm(768), torch.nn.Linear(768, text_embed_dim))
    t5 = t5.to(device)
    attn_mask = default(attn_mask, lambda: (token_ids != pad_id).long())

    t5.eval()

    with torch.no_grad():
        output = t5(input_ids = token_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()

    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, '... -> ... 1'), 0.) # just force all embeddings that is padding to be equal to 0.
    
    return encoded_text


def t5_encode_text(
    texts: List[str], 
    text_embed_dim = 8,
    name = DEFAULT_T5_NAME,
    return_attn_mask = False,
    device = "cuda:0"
):
    token_ids, attn_mask = t5_tokenize(texts, name = name)
    encoded_text = t5_encode_tokenized_text(token_ids, attn_mask = attn_mask, name = name, device = device, text_embed_dim = text_embed_dim)

    if return_attn_mask:
        attn_mask = attn_mask.bool()
        return encoded_text, attn_mask

    return encoded_text.to(device)

