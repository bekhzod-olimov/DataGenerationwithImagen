import torch, os, numpy as np, pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms as T
from utils import exists, SquarePad, insert_zeros

from utils import t5_encode_text

class CustomDataset(Dataset):
    
    def __init__(self, dataset, t5_encode_text, text_embed_dim, image_size, im_files = ['.jpg', '.jpeg', '.png', '.tiff']):
        super().__init__()
        
        self.image_size = image_size
        self.lps, self.ims = [], []
        
        if dataset == "lp":
        
            df = pd.read_csv(f"/home/ubuntu/workspace/bekhzod/imagen/lp_recognition_cropped/train/labels.csv")
            max_length = 8

            for i, name in enumerate(tqdm(df['filename'])):

                # if i == 50: break
                splitted = os.path.splitext(name)
                plate_num = splitted[0]
                im_file = splitted[1]
                assert im_file in im_files, f"Unsupported file type -> {name}"

                if name[0].isalpha() and name[1].isalpha(): break
                if "-" in name or " " in name: continue

                encoded_text = t5_encode_text(plate_num, text_embed_dim = text_embed_dim)

                if encoded_text.shape[0] != max_length:
                    temp = encoded_text.shape[0]
                    for _ in range(max_length - temp):
                        encoded_text = insert_zeros(encoded_text, [encoded_text.shape[0]])

                if encoded_text.flatten(0,1).shape[0] == 24:
                    self.lps.append(encoded_text.flatten(0,1))
                    self.ims.append(f"/home/ubuntu/workspace/bekhzod/imagen/lp_recognition_cropped/train/{df['filename'][i]}")
            
        elif dataset == "ocr":

            df = pd.read_csv(f"/home/ubuntu/workspace/dataset/banner_dataset/labelme_csv_ver1/train_data.csv")
            ints = np.random.randint(0, len(df['words']), size = 200)
            max_length = max([len(word) for word in df['words']])
            print("Obtaining text embeddings...")
            for i, word in tqdm(enumerate(df['words'])): # df['filename']
                if i in ints:
                # if i == 50: break
                    encoded_text = t5_encode_text(word, device='cuda:0', text_embed_dim = text_embed_dim)
                    if encoded_text.shape[0] != max_length:
                        temp = encoded_text.shape[0]
                        for _ in range(max_length - temp):
                            encoded_text = insert_zeros(encoded_text, [encoded_text.shape[0]])
                    # print(f"{word} -> {encoded_text.shape} -> {len(word)}")
                    if encoded_text.flatten(0,1).shape[0] != max_length * 3:
                        pad = torch.zeros(max_length, 1, text_embed_dim).to("cuda:0")
                        encoded_text = torch.cat([encoded_text, pad], dim = 1)    
                        print(encoded_text.shape)
                    self.lps.append(encoded_text.flatten(0,1))
                    self.ims.append(f"{df['filename'][i]}")
        
        self.transform = T.Compose([SquarePad(), T.Resize(image_size), T.ToTensor(), T.Lambda(lambda x: x[:3])])

    def __len__(self): return len(self.ims)

    def __getitem__(self, index): return self.transform(Image.open(self.ims[index])), self.lps[index]

# ds = CustomDataset(dataset = "ocr", t5_encode_text = t5_encode_text, text_embed_dim = 120, image_size = 128)