import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, RandomAffine, ColorJitter
from pprint import pprint


class VOCDataset(VOCDetection):
    def __init__(self, root='data', year=None, image_set=None,
                 download='True', transform=None, target_transform=None, transforms=None):
        # Initialize the parent class VOCDetection
        super().__init__(root, year, image_set, download, transform, target_transform, transforms)
        # Define category labels for the dataset
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train', 'tvmonitor']

    def __getitem__(self, item):
        # Get data from VOCDetection
        image, target = super().__getitem__(item)
        all_boxes = []
        all_labels = []
        # Extract bounding box coordinates and labels
        for obj in target["annotation"]["object"]:
            xmin = int(obj["bndbox"]["xmin"])
            ymin = int(obj["bndbox"]["ymin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymax = int(obj["bndbox"]["ymax"])

            all_boxes.append([xmin, ymin, xmax, ymax])
            all_labels.append(self.categories.index(obj["name"]))
        # Convert lists to tensors
        all_boxes = torch.FloatTensor(all_boxes)
        all_labels = torch.LongTensor(all_labels)
        target = {
            'boxes': all_boxes,
            'labels': all_labels
        }
        return image, target


if __name__ == '__main__':
    # Define image transformations
    transform = Compose([
        RandomAffine(degrees=(-5, 5),
                     translate=(0.15, 0.15),
                     scale=(0.85, 1.15),
                     shear=10
                     ),
        ColorJitter(brightness=0.123,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.5),
    ])
    # Initialize the dataset
    dataset = VOCDataset(root='/home/acer/PycharmProjects/ComputerVision/data/my_pascal_voc', year='2012',
                         transform=transform, image_set='train', download=False)
    # Retrieve and display the 200th image and its target
    image, target = dataset[200]
    image.show()