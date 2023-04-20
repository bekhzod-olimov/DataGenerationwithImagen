import os
from tqdm import tqdm

def train(trainer, t5_encode_text, max_bs, iterations, device, result_save_path, model_save_path, eval_freq, generate_freq, save_freq, text_embed_dim, save_name, text):
    
    text_embeds = t5_encode_text(text, text_embed_dim = text_embed_dim, device = device)
    os.makedirs(f"{result_save_path}/{save_name}", exist_ok = True)
    os.makedirs(f"{model_save_path}", exist_ok = True)

    for i in tqdm(range(iterations)):
        loss = trainer.train_step(unet_number = 1, max_batch_size = max_bs)

        if not (i % eval_freq):
            valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = max_bs)
            print(f"Validation loss for iteration {i} -> {valid_loss}\n")

        if not (i % generate_freq) and trainer.is_main: # is_main makes sure this can run in distributed
            # images = trainer.sample(batch_size = 1, texts = text, return_pil_images = True) 
            images = trainer.sample(batch_size = 1, text_embeds = text_embeds, return_pil_images = True) 
            images[0].save(f'{result_save_path}/{save_name}/sample_{i // generate_freq}.png')
            
        if not (i % save_freq) and i != 0:
            trainer.save(f"{model_save_path}/{save_name}_checkpoint_{i}.pt")
            
    
