import torch
from torchvision import transforms
from PIL import Image

from models import FeSVBiS

# ---------------- CONFIG ----------------
MODEL_PATH = "final_state_dict.pth"

class_names = [
    'basophil', 'eosinophil', 'erythroblast',
    'immature granulocyte', 'lymphocyte',
    'monocyte', 'neutrophil', 'platelet'
]

# ---------------- LOAD MODEL ----------------
def load_model():
    model = FeSVBiS(
        ViT_name="vit_base_r50_s16_224",
        num_classes=8,
        num_clients=6,
        in_channels=3,
        ViT_pretrained=False,
        initial_block=1,
        final_block=6,
        resnet_dropout=0.5,
        DP=False,
        mean=0,
        std=0
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, device

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ---------------- PREDICT ----------------
def predict_image(model, device, img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x, client_idx=0, chosen_block=6)
        probs = torch.softmax(logits, dim=1)

        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(torch.max(probs).item())

    return class_names[pred_idx], confidence