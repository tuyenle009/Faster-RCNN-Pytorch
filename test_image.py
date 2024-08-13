import cv2
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
import numpy as np


# Function to parse command line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Train faster RCNN model")
    parser.add_argument("--image_path", "-i", type=str, required=True, help="Path to the input image")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default="trained_models/best.pt",
                        help="Path to load the model checkpoint")
    parser.add_argument("--conf_threshold", "-c", type=float, default=0.3, help="Confidence threshold for predictions")
    args = parser.parse_args()
    return args


# Function to test the model with given arguments
def test(args):
    # List of categories for object detection
    categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                  'train', 'tvmonitor']

    # Create the device element, use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    model = fasterrcnn_mobilenet_v3_large_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the box predictor with a new one for our number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels, num_classes=21)

    # Load the model from the checkpoint
    checkpoint = torch.load(args.saved_checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Load the input image
    ori_image = cv2.imread(args.image_path)
    image = cv2.imread(args.image_path)

    # Convert image from BGR to RGB and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1)) / 255.0
    image = [torch.from_numpy(image).float().to(device)]

    # Set the model to evaluation mode and disable gradient computation
    model.eval()
    with torch.no_grad():
        output = model(image)[0]

        # Extract bounding boxes, labels, and scores
        bboxes = output["boxes"]
        labels = output["labels"]
        scores = output["scores"]

        # Iterate through predictions and draw boxes on the image
        for bbox, label, score in zip(bboxes, labels, scores):
            if score.item() > args.conf_threshold:
                xmin, ymin, xmax, ymax = bbox
                cv2.rectangle(ori_image, (int(xmin), int(ymin), int(xmax), int(ymax)), (0, 0, 255), 3)
                category = categories[label]

                # Write category text on the image
                cv2.putText(ori_image, category, (int(xmin), int(ymin)), cv2.FONT_ITALIC, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

        # Save the resulting image with predictions
        cv2.imwrite("prediction.jpg", ori_image)


# Main function to parse arguments and call the test function
if __name__ == '__main__':
    args = get_args()
    test(args)