from typing import List
from transformers import T5Tokenizer, T5EncoderModel, T5Config
from einops import rearrange
import torch, numpy as np, PIL.Image as Image
import torchvision.transforms.functional as F

DEFAULT_T5_NAME = "google/t5-v1_1-base"
T5_CONFIGS = {}
MAX_LENGTH = 256

def tn_to_np(im): return (im * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def save_ims(gen_ims, save_path):
    
    for idx, gen_im in enumerate(gen_ims):
        im = tn_to_np(gen_im[0])
        Image.fromarray(im).save(f"{save_path}/gen_im_{idx}.jpg")

def exists(val): return val is not None


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w)/2)
        hp_rem = (max_wh - w)%2
        vp = int((max_wh - h)/2)
        vp_rem = (max_wh - h)%2
        padding = (hp, vp, hp+hp_rem, vp+vp_rem)
        return F.pad(image, padding, 255, 'constant')


def insert_zeros(x, all_j):
    zeros_ = torch.zeros_like(x[:1])
    pieces = []
    i      = 0
    for j in all_j + [len(x)]:
        pieces.extend([x[i:j],
                       zeros_])
        i = j

    return torch.cat(pieces[:-1],
                      dim=0     )

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def get_model_and_tokenizer(name):
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']

def get_model(name):
    model = T5EncoderModel.from_pretrained(name)
    return model


def t5_tokenize(
    texts: List[str],
    name = DEFAULT_T5_NAME
):
    t5, tokenizer = get_model_and_tokenizer(name)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = 'longest',
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)
    return input_ids, attn_mask

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

