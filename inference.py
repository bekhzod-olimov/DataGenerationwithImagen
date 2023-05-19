# Import libraries
import argparse, os
from model import get_imagen
from imagen_pytorch import ImagenTrainer
from PIL import Image
from utils import tn_to_np, t5_encode_text, save_ims

def run(args):

    # Create directory to save results
    os.makedirs(f"{args.save_results_path}", exist_ok = True)
    print("Initializing a trainer...")
    
    # Get Imagen model
    imagen = get_imagen(args.text_embed_dim, args.im_size)
    
    # Initialize trainer object
    trainer = ImagenTrainer(imagen = imagen, split_valid_from_train = True).to(args.device)
    print("The trainer is initialized! Loading checkpoint...")
    
    # Load model pretrained checkpoint from local dir
    trainer.load(args.saved_model_path)

    print("Starting inference process...")
    
    # Get and save generated images
    gen_ims = [trainer.sample(text_embeds = t5_encode_text(f"{text}", text_embed_dim = args.text_embed_dim).to(args.device).flatten(0,1).unsqueeze(0), cond_scale = 3.) for text in args.texts]
    save_ims(gen_ims, f"{args.save_results_path}")
    print("Inference is done!")
    
if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Imagen Inference Arguments")
    
    # Add arguments
    parser.add_argument("-r", "--texts", type = list, default = ["미래는 있다", "최대 3년", "효과가 있습니다", "집으로 보내줍니다", "주택 및 빌라", "최소 3%까지", "산업단지", "현수막", "지하철", "그 시도는 부분적인 성공밖에 거두지 못했다.", "종종 부분적인 색채 주위에 외곽선이 사용되었다.", "왜냐하면, 도둑들이 이들 산삼을 쉽게 찾아낸다는 부분적인 이유로 이를 보호 식물로 지정해놓았기 때문입니다.", "이것은 편미분 방정식이다", "그 외에는 부분식으로 알려져있습니다.", "전도연은 2007칸느 영화제를 포함해 8번의 여우주연상을 받았어.", "최우수 여우주연상은 내 친구가 탔어.", "그녀는 아카데미 여우주연상을 두 번 수상했다.", "끈을 자르지 말고 매듭을 풀어라.", "그가 그녀의 팔을 묶고 있던 끈을 풀어 주었다.", "첫째, 다음에는 혼자 걷지 말아라.", "하지만 여러분은 안전을 위해 몇 가지를 기억해야만 합니다; 결코 혼자 걷지 마세요."], help = "Text prompts list")
    # parser.add_argument("-r", "--texts", type = list, default = ["102가3456", "98마2478", "407오7895", "158로1007", "인천89바4871", "08사0793", "84모4711", "456바6908", "37보3564", "71모2471", "부산01다8713", "32다2312", "03고0077", "경기89바4871", "535머6006", "88호0497", "75주6845", "대구56마0017", "57어2897", "19누3471", "074구0793", "24오1788", "517두6974", "57너3564", "87로5755", "13소1481", "24자1789", "경기31가8741", "144육4785"], help = "Text prompts list")
    parser.add_argument("-te", "--text_embed_dim", type = int, default = 120, help = "Text embedding size")
    parser.add_argument("-is", "--im_size", type = int, default = 128, help = "Input and output image size")
    parser.add_argument("-d", "--device", type = str, default = "cuda:0", help = "GPU device number")
    parser.add_argument("-sm", "--saved_model_path", type = str, default = "saved_models/ocr_checkpoint_1200000.pt", help = "Path to the directory with the trained/saved model")
    parser.add_argument("-sr", "--save_results_path", type = str, default = "inference_generated_samples", help = "Path to the directory to save the inference results")
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args)
