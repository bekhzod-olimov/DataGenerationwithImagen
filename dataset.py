# Import libraries
import torch, os, numpy as np, pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms as T
from utils import exists, SquarePad, insert_zeros, t5_encode_text

class CustomDataset(Dataset):
    
    """
    
    This class gets several parameters and returns dataset to generate images.
    
    Parameters:
    
        dataset          - dataset name, str;
        t5_encode_text   - a model to encode text, torch model object;
        text_embed_dim   - number of dimensions of the text to be embedded, int;
        image_size       - size of image in the dataset, int;
        im_files         - valid image file extensions, list -> str.
    
    """
    
    def __init__(self, dataset, t5_encode_text, text_embed_dim, image_size, im_files = ['.jpg', '.jpeg', '.png', '.tiff']):
        super().__init__()
        
        # Initialize transformations to be applied
        self.transform = T.Compose([SquarePad(), T.Resize(image_size), T.ToTensor(), T.Lambda(lambda x: x[:3])])
        self.lps, self.ims = [], []
        
        # For lp data
        if dataset == "lp":
        
            # Read csv data
            df = pd.read_csv(f"/home/ubuntu/workspace/bekhzod/imagen/lp_recognition_cropped/train/labels.csv")
            
            # Set maximum length of text data
            max_length = 8

            # Go through text data
            for i, name in enumerate(tqdm(df["filename"])):

                # Get filename and extension
                splitted = os.path.splitext(name)
                plate_num, im_file = splitted[0], splitted[1]
                
                # Assertion
                assert im_file in im_files, f"Unsupported file type -> {name}"

                # Break the loop if the first two characters are alphabet letters
                if name[0].isalpha() and name[1].isalpha(): break
                # Continue when special characters are in the text
                if "-" in name or " " in name: continue

                # Get embedded text
                encoded_text = t5_encode_text(plate_num, text_embed_dim = text_embed_dim)

                # In the case the text embedding is not equal to maximum length
                if encoded_text.shape[0] != max_length:
                    temp = encoded_text.shape[0]
                    for _ in range(max_length - temp):
                        # Pad with zeros
                        encoded_text = insert_zeros(encoded_text, [encoded_text.shape[0]])

                # Flatten text embeddings and add to the pre-defined lists
                if encoded_text.flatten(0,1).shape[0] == 24:
                    self.lps.append(encoded_text.flatten(0,1))
                    self.ims.append(f"/home/ubuntu/workspace/bekhzod/imagen/lp_recognition_cropped/train/{df['filename'][i]}")
            
        # For ocr data
        elif dataset == "ocr":

            # Read csv data
            df = pd.read_csv(f"/home/ubuntu/workspace/dataset/banner_dataset/labelme_csv_ver1/train_data.csv")
            
            # Get random integers
            ints = np.random.randint(0, len(df['words']), size = 200)
            # Set maximum length of text data
            max_length = max([len(word) for word in df['words']])
            print("Obtaining text embeddings...")
            
            # Go through text data
            for i, word in tqdm(enumerate(df["words"])): 
                if i in ints:
                # if i == 50: break
                    # Get embedded text
                    encoded_text = t5_encode_text(word, device='cuda:0', text_embed_dim = text_embed_dim)
                    # In the case the text embedding is not equal to maximum length
                    if encoded_text.shape[0] != max_length:
                        temp = encoded_text.shape[0]
                        for _ in range(max_length - temp):
                            encoded_text = insert_zeros(encoded_text, [encoded_text.shape[0]])
                    # print(f"{word} -> {encoded_text.shape} -> {len(word)}")
                    if encoded_text.flatten(0,1).shape[0] != max_length * 3:
                        # Pad with zeros
                        pad = torch.zeros(max_length, 1, text_embed_dim).to("cuda:0")
                        encoded_text = torch.cat([encoded_text, pad], dim = 1)    
                    # Add to the pre-defined lists
                    self.lps.append(encoded_text.flatten(0,1))
                    self.ims.append(f"{df['filename'][i]}")
        
    # Get length of the dataset
    def __len__(self): return len(self.ims)

    # Return tranformed image and its corresponding text embedding
    def __getitem__(self, idx): return self.transform(Image.open(self.ims[idx])), self.lps[idx]

# ds = CustomDataset(dataset = "ocr", t5_encode_text = t5_encode_text, text_embed_dim = 120, image_size = 128)
