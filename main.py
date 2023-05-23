# Import libraries
from imagen_pytorch import ImagenTrainer
from model import get_imagen
import argparse, torch
from dataset import CustomDataset
from utils import t5_encode_text
from train import train

def run(args):
    
    """
    
    This functions gets arguments and runs the script.
    
    Parameter:
    
        args - parsed arguments, argparse object.
        
    """
    
    # Initialize sample text based on dataset type
    text = "123나0456" if args.data == "lp" else "학교"
    print("Extracting data...\n")
    
    # Get dataset
    ds = CustomDataset(dataset = args.data, t5_encode_text = t5_encode_text, text_embed_dim = args.text_embed_dim, image_size = args.im_size)
    print("Data extraction is done!\n")
    
    # Initialize imagen and trainer 
    print("Constructing model and trainer...\n")
    imagen = get_imagen(args.text_embed_dim, args.im_size)
    trainer = ImagenTrainer(imagen = imagen, split_valid_from_train = True).to(args.device)
    print("Trainer is ready for train!\n")
    
    # Add datasets to the trainer
    trainer.add_train_dataset(ds, batch_size = args.batch_size)
    
    # Pretrained checkpoint
    if args.pretrained is not None:
        trainer.load(args.pretrained)
        print(f"Pretrained model from {args.pretrained} is successfully loaded!")
    
    # Start training with the pre-defined arguments
    train(trainer, t5_encode_text, args.batch_size, 
          args.iterations, device = args.device,
          result_save_path = args.save_results_path, 
          model_save_path = args.save_model_path,
          eval_freq = args.eval_freq,
          generate_freq = args.generate_freq, 
          save_freq = args.save_freq, 
          text_embed_dim = args.text_embed_dim, 
          save_name = args.data, text = text)
    
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Imagen Train Arguments")
    
    # Add arguments
    parser.add_argument("-dt", "--data", type = str, default = "ocr", help = "Path to the data")
    parser.add_argument("-bs", "--batch_size", type = int, default = 8, help = "Mini-batch size")
    parser.add_argument("-te", "--text_embed_dim", type = int, default = 120, help = "Text embedding size")
    parser.add_argument("-pt", "--pretrained", type = str, default = "path/to/checkpoint", help = "Fine tuning using a pretrained model") # /home/ubuntu/workspace/bekhzod/imagen/saved_models/ocr_1000000.pt
    parser.add_argument("-is", "--im_size", type = int, default = 128, help = "Input and output image size")
    parser.add_argument("-i", "--iterations", type = int, default = 120010, help = "Train iterations number")
    parser.add_argument("-d", "--device", type = str, default = 'cuda:0', help = "GPU device number")
    parser.add_argument("-ef", "--eval_freq", type = int, default = 500, help = "Evaluation frequency (iterations)")
    parser.add_argument("-gf", "--generate_freq", type = int, default = 5000, help = "Generation frequency (iterations)")
    parser.add_argument("-sf", "--save_freq", type = int, default = 60000, help = "Model save frequency (iterations)")
    parser.add_argument("-sm", "--save_model_path", type = str, default = "saved_models", help = "Path to the directory to save a trained model")
    parser.add_argument("-sr", "--save_results_path", type = str, default = "train_generated_samples", help = "Path to the directory to save the train results")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args)
