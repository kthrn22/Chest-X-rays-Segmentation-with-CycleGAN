import torch
from PIL import Image
from generator import *
from discriminator import *
from torchvision import transforms
from torchvision.utils import save_image
from test_options import *
import os

checkpoint = torch.load("./checkpoint.pt")

checkpoint.keys()

generator = Generator(1, 6).to("cuda")

Config = TestOptions().gather_options()

if Config.generator_type == "mask":
    generator.load_state_dict(checkpoint['generator_zebra'])
if Config.generator_type == "x_ray":
    generator.load_state_dict(checkpoint['generator_horse'])

test_image_folder = os.listdir(Config.test_image_folder)

def translate_image(root, root_translated, image_path, generator):
    transformation = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    file_id = image_path.split(".")[0]
    image = transformation(Image.open(os.path.join(root, image_path)).convert("L")).view(1, 1, 128, 128).to("cuda")
    translated_image = generator(image)
    translated_image = translated_image.detach().cpu()
    save_image(translated_image, os.path.join(root_translated, "{}_translated.png".format(file_id)))

for index in range(len(test_image_folder)):
    translate_image(Config.test_image_folder, Config.generated_image_folder, test_image_folder[index], generator)
