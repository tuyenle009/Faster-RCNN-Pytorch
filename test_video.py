import cv2
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
import numpy as np


# Parse command line arguments
def get_args():
    parser = argparse.ArgumentParser(description="Run Faster R-CNN on video")
    parser.add_argument("--video_path", "-v", type=str, required=True, help="Path to the input video")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default="trained_models/best.pt",
                        help="Model checkpoint path")
    parser.add_argument("--conf_threshold", "-c", type=float, default=0.3, help="Confidence threshold")
    return parser.parse_args()


# Run object detection on video
def test(args):
    # Categories for object detection
    categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                  'train', 'tvmonitor']

    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load checkpoint
    model = fasterrcnn_mobilenet_v3_large_fpn()
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels, num_classes=21)
    model.load_state_dict(torch.load(args.saved_checkpoint)["model_state_dict"])
    model.to(device)

    # Video capture and writer setup
    cap = cv2.VideoCapture(args.video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = [torch.from_numpy(image).float().to(device)]

        # Model inference
        model.eval()
        with torch.no_grad():
            output = model(image)[0]
            bboxes, labels, scores = output["boxes"], output["labels"], output["scores"]

            # Draw predictions on frame
            for bbox, label, score in zip(bboxes, labels, scores):
                if score.item() > args.conf_threshold:
                    xmin, ymin, xmax, ymax = map(int, bbox)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                    category = categories[label]
                    cv2.putText(frame, category, (xmin, ymin), cv2.FONT_ITALIC, 1, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(frame)

    cap.release()
    out.release()

# Entry point
if __name__ == '__main__':
    args = get_args()
    test(args)