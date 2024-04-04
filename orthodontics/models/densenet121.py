import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class OrthodonticsDenseNet121(nn.Module):
    def __init__(self):
        super(OrthodonticsDenseNet121, self).__init__()
        # Load the pre-trained DenseNet121 model
        self.densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # Replace the classifier with one that has a single output feature
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_features, 1)
    
    def forward(self, x):
        x = self.densenet121(x)
        # Apply sigmoid to ensure output is between 0 and 1
        return torch.sigmoid(x)
    
class OrthodonticsDenseNet121WithLabels(nn.Module):
    def __init__(self, num_labels):
        super(OrthodonticsDenseNet121WithLabels, self).__init__()
        # Load the pre-trained DenseNet121 model
        self.densenet121 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # Remove the classifier layer
        self.features = self.densenet121.features
        self.densenet121.classifier = nn.Identity()  # Set the classifier to an identity operation

        # Additional layers
        self.label_embedding = nn.Embedding(num_labels, 10)  # Embed labels into a 10-dimensional space
        # Adjust combined_processor to match the total feature size (1024 from DenseNet + 10 from label embedding)
        self.combined_processor = nn.Linear(1024 + 10, 1)

    def forward(self, x, labels):
        x = self.features(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # Flatten the features to match [batch_size, num_features]
        
        labels = self.label_embedding(labels)
        combined_features = torch.cat([x, labels], dim=1)  # Concatenate along the feature dimension
        output = self.combined_processor(combined_features)
        return torch.sigmoid(output)