import torch
import json
import os
from PIL import Image
from torchvision import transforms
from src.perception.model import FruitDetection

class FruitClassifier:
    """
    Inference wrapper for the Agri-Swarm fruit classification model.
    This class handles model loading, image preprocessing, and prediction.
    """
    def __init__(self, model_path="models/fruit_classifier.pth", 
                 stats_path="data/processed/dataset_stats.json",
                 device=None):
        # 1. Set Device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Load Dataset Statistics (Agnostic approach)
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Statistics file not found at {stats_path}. Please run prepare_data.py first.")
            
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        self.mean = stats['mean']
        self.std = stats['std']
        self.img_size = tuple(stats['img_size'])
        
        # 3. Load Class Names (Assuming alphabetical order from ImageFolder during training)
        # Note: In a production environment, saving class_names in the JSON is safer.
        # For now, we manually define or ensure they match the training folders.
        self.class_names = ['apple_fresh', 'apple_rotten', 'banana_fresh', 'banana_rotten', 'orange_fresh', 'orange_rotten']

        # 4. Initialize and Load Model
        # We pass N_input and M_input to trigger the dynamic flatten calculation
        self.model = FruitDetection(
            num_classes=len(self.class_names),
            N_input=self.img_size[0],
            M_input=self.img_size[1]
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set to evaluation mode (disables Dropout/BatchNorm updates)
        
        # 5. Define Preprocessing Pipeline
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def predict(self, image_input):
        """
        Predicts the class and confidence of a given image.
        :param image_input: Can be a path (str) or a PIL Image object.
        :return: (predicted_class_name, confidence_score)
        """
        # Load image if a path is provided
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input

        # Preprocess and add batch dimension [1, C, H, W]
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            # Apply Softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        class_name = self.class_names[predicted_idx.item()]
        score = confidence.item()

        return class_name, score

# --- Example Usage for Testing ---
if __name__ == "__main__":
    # This block allows testing the module independently
    try:
        classifier = FruitClassifier()
        # Replace with an actual image path from your data/raw/test folder
        test_img = "data/raw/test/apple/vertical_flip_Screen Shot 2018-06-08 at 5.13.02 PM.png" 
        
        if os.path.exists(test_img):
            label, conf = classifier.predict(test_img)
            print(f"✅ Prediction: {label} (Confidence: {conf:.2f})")
        else:
            print("⚠️ Provide a valid image path to test inference.")
    except Exception as e:
        print(f"💥 Inference error: {e}")