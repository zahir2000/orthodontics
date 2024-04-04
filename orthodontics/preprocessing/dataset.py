from torch.utils.data import Dataset

class OrthodonticsDataset(Dataset):
    def __init__(self, X, y, transform=None):
        """
        Args:
            X (numpy.ndarray): Numpy array of shape (N, H, W, C) containing the images,
                               where N is the number of images, H is the height,
                               W is the width, and C is the number of channels.
            y (numpy.ndarray): Numpy array containing the labels/targets for each image.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert the image data to a PIL Image to apply transformations
        image = self.X[idx]
        # If your images are not already in PIL Image format or a tensor, uncomment the next line
        # image = Image.fromarray(image)
        
        label = self.y[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
class OrthodonticsDatasetWithLabels(Dataset):
    """
    A custom dataset class that includes labels.

    Args:
        X (list): List of input data.
        y (list): List of target labels.
        labels (list): List of encoded labels.
        transform (callable, optional): Optional transform to be applied to the input data.

    Attributes:
        X (list): List of input data.
        y (list): List of target labels.
        labels (list): List of encoded labels.
        transform (callable, optional): Optional transform to be applied to the input data.
    """

    def __init__(self, X, y, labels, transform=None):
        self.X = X
        self.y = y
        self.labels = labels  # Store encoded labels
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        rotation = self.y[idx]
        label = self.labels[idx]  # Encoded label

        if self.transform:
            image = self.transform(image)

        # Now also return the encoded label
        return image, rotation, label