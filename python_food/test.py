from models import myresnet
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from mydataset import Nutrition
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from collections import OrderedDict

test_transform = transforms.Compose([
    transforms.Resize((270, 480), Image.BILINEAR),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

data_root = r"D:\Food_Calorie_Prediction\nutrition5k\nutrition5k_dataset"

nutrition_rgbd_ims_root = os.path.join(data_root, 'imagery')
nutrition_test_txt = os.path.join(data_root, 'imagery', 'rgb_in_overhead_test_processed.txt')

testset = Nutrition(nutrition_rgbd_ims_root, nutrition_test_txt, transform=test_transform)

test_loader = DataLoader(testset,
                         batch_size=4,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True
                         )

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

net_rgb = myresnet.resnet101(rgbd=False)

checkpoint_path = r"D:\Food_Calorie_Prediction\python_food\saved\new\regression_nutrition_rgb_resnet101_editname\ckpt_best.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

new_state_dict = OrderedDict()
for k, v in checkpoint['net'].items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

net_rgb.load_state_dict(new_state_dict)
net_rgb.to(device)
net_rgb.eval()

criterion = nn.SmoothL1Loss()

epoch_iterator = tqdm(test_loader,
                      desc="Testing... (loss=X.X)",
                      bar_format="{l_bar}{r_bar}",
                      dynamic_ncols=True)

test_loss = 0
calories_loss = 0
mass_loss = 0
fat_loss = 0
carb_loss = 0
protein_loss = 0

with torch.no_grad():
    for batch_idx, x in enumerate(epoch_iterator):
        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()

        outputs = net_rgb(inputs)

        calories_total_loss = total_calories.shape[0] * criterion(outputs[0], total_calories) / total_calories.sum().item()
        mass_total_loss = total_calories.shape[0] * criterion(outputs[1], total_mass) / total_mass.sum().item()
        fat_total_loss = total_calories.shape[0] * criterion(outputs[2], total_fat) / total_fat.sum().item()
        carb_total_loss = total_calories.shape[0] * criterion(outputs[3], total_carb) / total_carb.sum().item()
        protein_total_loss = total_calories.shape[0] * criterion(outputs[4], total_protein) / total_protein.sum().item()

        loss = calories_total_loss + mass_total_loss + fat_total_loss + carb_total_loss + protein_total_loss

        test_loss += loss.item()
        calories_loss += calories_total_loss.item()
        mass_loss += mass_total_loss.item()
        fat_loss += fat_total_loss.item()
        carb_loss += carb_total_loss.item()
        protein_loss += protein_total_loss.item()

        epoch_iterator.set_description(
            "loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f " % (
                test_loss / (batch_idx + 1),
                calories_loss / (batch_idx + 1),
                mass_loss / (batch_idx + 1),
                fat_loss / (batch_idx + 1),
                carb_loss / (batch_idx + 1),
                protein_loss / (batch_idx + 1)
            )
        )

print(f"\nTest finished! Final loss: {test_loss / len(test_loader):.5f}")