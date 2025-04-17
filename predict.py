# predict.py
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from models.vqa_model import VQAModel
import yaml

# Load config
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load model
model = VQAModel(config)
model.load_state_dict(torch.load("vqa_model.pth", map_location=torch.device('cpu')))  # or 'cuda'
model.eval()

# Load and preprocess image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dim

# Preprocess question
def preprocess_question(question):
    tokens = tokenizer(question, return_tensors='pt', padding='max_length', max_length=32, truncation=True)
    return tokens

def predict(image_path, question):
    image_tensor = preprocess_image(image_path)
    question_tokens = preprocess_question(question)

    input_ids = question_tokens["input_ids"]
    attention_mask = question_tokens["attention_mask"]

    with torch.no_grad():
        logits = model(image_tensor, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1)

        # Use fixed label mapping (based on training logic)
        label_map = {0: "no", 1: "yes"}
        return label_map[predicted.item()]

# Example usage
if __name__ == "__main__":
    img_path = "ecgImages/ECG00394_clinical_512.png"  # replace with your image
    question = "Does this ECG show symptoms of non-diagnostic t abnormalities?"

    output = predict(img_path, question)
    print(f"Prediction: {output}")
