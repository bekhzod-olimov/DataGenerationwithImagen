# Import libraries
import os
from tqdm import tqdm

def train(trainer, t5_encode_text, max_bs, iterations, device, result_save_path, model_save_path, eval_freq, generate_freq, save_freq, text_embed_dim, save_name, text):
    
    # Get text embeddings to sample images
    text_embeds = t5_encode_text(text, text_embed_dim = text_embed_dim, device = device)
    # Create directories to save model and results
    os.makedirs(f"{result_save_path}/{save_name}", exist_ok = True); os.makedirs(f"{model_save_path}", exist_ok = True)

    # Start training
    for i in tqdm(range(iterations)):
        
        # Compute train loss
        loss = trainer.train_step(unet_number = 1, max_batch_size = max_bs)

        # Compute validation loss
        if not (i % eval_freq):
            valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = max_bs)
            print(f"Validation loss for iteration {i} -> {valid_loss}\n")

        # Sample images 
        if not (i % generate_freq) and trainer.is_main:
            images = trainer.sample(batch_size = 1, text_embeds = text_embeds, return_pil_images = True) 
            images[0].save(f'{result_save_path}/{save_name}/sample_{i // generate_freq}.png')
            
        # Save a trained model
        if not (i % save_freq) and i != 0: trainer.save(f"{model_save_path}/{save_name}_checkpoint_{i}.pt")
