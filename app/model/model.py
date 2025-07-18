import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

model_path="model/best_model_unet.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class UNetAutoencoder(nn.Module):
    def __init__(self):
        super(UNetAutoencoder, self).__init__()

        self.enc1 = self.conv_block(1, 64)     
        self.enc2 = self.conv_block(64, 128)   
        self.enc3 = self.conv_block(128, 256)  
        self.enc4 = self.conv_block(256, 512)  
        self.enc5 = self.conv_block(512, 1024) #bottlenect

        self.up1 = self.up_block(1024, 512)    # 32 -> 64
        self.up2 = self.up_block(1024, 256)    # 64 -> 128 (concat with enc4)
        self.up3 = self.up_block(512, 128)     # 128 -> 256 (concat with enc3)
        self.up4 = self.up_block(256, 64)      # 256 -> 512 (concat with enc2)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d1 = self.up1(e5)
        d1 = torch.cat([d1, e4], dim=1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d4 = self.up4(d3)
        d4 = torch.cat([d4, e1], dim=1)

        out = self.final(d4)
        return out

# Load your trained model
model = UNetAutoencoder()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def predict_pipeline(image):
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        prediction=model(input_tensor)
        prediction_np = prediction.detach().cpu().squeeze().numpy()
        
        prediction_np=np.round((prediction_np+1)*255)//2
        prediction_np=prediction_np.astype(np.uint8)
        output=Image.fromarray(prediction_np)
    return output