import torch
import torch.nn as nn

class FruitDetection(nn.Module):
    def __init__(self, num_classes, num_channels=3, N_input=100, M_input=100):
        # Initialize the parent nn.Module class
        super(FruitDetection,self).__init__()
        self.N_input = N_input
        self.M_input = M_input
        
        # nn.Sequential groups the layers together, making the forward pass cleaner and easier to manage
        self.cnnLayers = nn.Sequential(
            # First convolutional layer
            # Detects low-level features such as edges and textures
            # Input: RGB images (3 channels), Output: 16 feature maps, Kernel size: 3x3, Padding: 1 to maintain spatial dimensions
            nn.Conv2d(num_channels, 16, kernel_size=3, padding=1),
            # Batch normalization stabilizes training by re-centering and re-scaling layer inputs, improving convergence speed
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            # Max pooling layer halves the spatial dimensions of the image (extracting dominant features)
            nn.MaxPool2d(2),
            
            # Second convolutional layer
            # Detects more complex features by building on the low-level features from the first layer
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third convolutional layer
            # Detects even more complex features, such as shapes and patterns specific to fruits
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # ReLU activation function is applied to introduce non-linearity, allowing the model to learn complex patterns.
        self.relu = nn.ReLU()

        # AUTOMATIC AND AGNOSTIC CALCULATION of space for the linear layer
        # This dynamically computes how many features exit the CNN block, ensuring no shape mismatch errors
        self.flatten_size = self._get_conv_output_size(num_channels, N_input, M_input)

        # Fully connected (Dense) layers
        # Takes the flattened 1D representation of the CNN feature maps and maps them to the final class probabilities
        self.fc1 = nn.Linear(self.flatten_size, 256)
        
        # Dropout randomly zeroes out 50% of the neurons during training to prevent the network from memorizing the data (Overfitting)
        self.dropout = nn.Dropout(0.5) 
        
        # Final Output layer where the number of neurons equals the number of target classes
        self.fc2 = nn.Linear(256, num_classes) 

    def _get_conv_output_size(self, c, h, w):
        """Helper to calculate the exact size after convolutions."""
        # Using torch.no_grad() because we don't need to track gradients for this spatial calculation
        with torch.no_grad():
            # Create a dummy image tensor with the expected input dimensions
            dummy_input = torch.zeros(1, c, h, w)
            # Pass it through the convolutional layers
            dummy_output = self.cnnLayers(dummy_input)
            # Return the total number of elements in the output, which is the exact size needed for the Linear layer.
            return dummy_output.numel() 

    def forward(self, x):
        # 1. Feature Extraction: Pass the input through all Convolutional and Pooling layers
        x = self.cnnLayers(x)

        # 2. Flattening: Transform the 4D tensor [batch, channels, height, width] into a 2D tensor [batch, features]
        # x.size(0) preserves the dynamic batch size, while -1 infers the rest of the elements
        x = x.view(x.size(0), -1)
        
        # 3. Decision Making: Pass through the first fully connected layer and apply ReLU
        x = self.relu(self.fc1(x))

        # 4. Regularization: Apply dropout (active only during training)
        x = self.dropout(x) 
        
        # 5. Output classification logits (Raw scores before Softmax/CrossEntropy)
        x = self.fc2(x) 
        return x