import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Importing model and data loader
from src.perception.model import FruitDetection
from src.perception.data_loader import get_data_loaders

def plot_confusion_matrix(true_labels, predictions, class_names, save_path="models/confusion_matrix.png"):
    """
    Generate and save a heatmap of the confusion matrix.
    The confusion matrix helps visualize the performance of the classification model,
    showing exactly where the model is confusing two classes (e.g., Ripe Apple vs Rotten Apple).
    """
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    # seaborn.heatmap renders the numerical matrix into a color-coded grid
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Free up memory by closing the plot
    print(f"📊 Confusion matrix saved in: {save_path}")

# Training function
def train_model():
    # Set random seeds for reproducibility
    # This ensures that every time you run this script, the random weights initialization
    # and dataset shuffling produce the exact same results. Crucial for scientific debugging.
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Automatically select the GPU if available, otherwise fallback to CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "models/fruit_classifier.pth"
    CM_SAVE_PATH = "models/confusion_matrix.png"
    DATA_DIR = "data/raw"

    # Make sure the models folder exists to avoid FileNotFoundError during saving
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Hyperparameters defining the training process
    EPOCHS = 15
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
  
    # Get data loaders, class names, and image size from the data loader function
    train_loader, val_loader, test_loader, class_names, img_size = get_data_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE)

    num_classes = len(class_names) # Dynamically set based on dataset folders

    img_width, img_height = img_size
    num_channels = 3 # RGB images

    # Initialize the model, loss function, and optimizer
    # Move the model to the target DEVICE (GPU/CPU)
    model = FruitDetection(num_classes=num_classes, num_channels=num_channels, N_input=img_width, M_input=img_height).to(DEVICE)
    
    # CrossEntropyLoss is the standard loss function for multi-class classification tasks
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer adapts the learning rate for each parameter, generally converging faster than standard SGD
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_accuracy = 0.0 # Variable to track the best validation accuracy

    # Logging dataset and model properties
    print(f"\n---🚀 Starting training on {DEVICE} ---")
    print(f"📊 Number of classes: {num_classes}")
    print(f"🖼️ Image size: {img_size}")
    print(f"📂 Training samples: {len(train_loader.dataset)}")
    print(f"📂 Validation samples: {len(val_loader.dataset)}")
    print(f"📂 Test samples: {len(test_loader.dataset)}")
    print("-" * 60)


    # ==========================================
    # 1. Training loop with validation
    # ==========================================
    for epoch in range(EPOCHS):
        # --- Training phase ---
        # model.train() enables layers like Dropout and updates BatchNorm running statistics
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            # Transfer data to the same device as the model
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad() # Clear gradients from the previous step
            outputs = model(images) # Forward pass: compute predicted outputs
            
            # Calculate the loss (difference between predictions and actual labels)
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step to update model weights
            optimizer.step()

            # Accumulate loss and calculate accuracy for the current batch
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # Get the index of the highest probability
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Average loss and accuracy for the entire training epoch
        train_loss = running_train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # --- Validation phase ---
        # model.eval() turns off Dropout and uses fixed statistics for BatchNorm
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        # torch.no_grad() disables gradient calculation, saving Memory and CPU/GPU compute time during inference
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        # --- EPOCH LOGGING ---
        print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # --- SAVING THE BEST MODEL ---
        # Only save the weights if the model performs better on the validation set than previous epochs
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # Save the state_dict (which contains only the trainable parameters/weights, not the whole class)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"   ⭐ Model improved! Saving weights to {SAVE_PATH}...")

    print("-" * 60)
    print(f"✅ Training completed! Best Val Acc: {best_accuracy:.2f}%")
    print("\n--- 🔬 Starting Evaluation on Test Set ---")

    # ==========================================
    # 2. FINAL TEST AND CONFUSION MATRIX
    # ==========================================
    
    # Load the best weights saved during the validation phase
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval() # Ensure the model is in evaluation mode
    
    correct_test = 0
    total_test = 0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            # Save predictions and actual labels for the confusion matrix
            # .cpu().numpy() moves the tensor back to system memory and converts it for Scikit-Learn compatibility
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    test_acc = 100 * correct_test / total_test
    print(f"🏆 Final Accuracy on Test Set: {test_acc:.2f}%\n")

    # Generate the graphical Confusion Matrix
    plot_confusion_matrix(all_true_labels, all_predictions, class_names, save_path=CM_SAVE_PATH)

if __name__ == "__main__":
    train_model()