import torch
from src.voc_dataset import VOCDataset
from torchvision.transforms import ToTensor, Compose, Normalize, RandomAffine, ColorJitter
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from tqdm.autonotebook import tqdm
import argparse
import os.path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
import shutil


# create collate function
def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

def get_args():
    parser = argparse.ArgumentParser(description="Train faster rcnn model")
    parser.add_argument("--data_path", "-d", type=str, default="data", help="Path to dataset")
    parser.add_argument("--year", "-y", type=str, default="2012")
    parser.add_argument("--num_epochs", "-n", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--momentum", "-m", type=float, default=0.9, help="Momentum for optimizer")
    parser.add_argument("--log_folder", "-p", type=str, default="tensorboard",
                        help="Path to generated tensorboard")
    parser.add_argument("--checkpoint_folder", "-c", type=str, default="trained_models",
                        help="Path to generated tensorboard")
    parser.add_argument("--saved_checkpoint", "-o", type=str, default=None, help="Continue from this checkpoint")
    args = parser.parse_args()
    return args


def train(args):
    # create the device element "CPU"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transform inputs
    train_transform = Compose([
        RandomAffine(degrees=(-5, 5),
                     translate=(0.15, 0.15),
                     scale=(0.85, 1.15),
                     shear=10
                     ),
        ColorJitter(brightness=0.123,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.5),
        ToTensor(),  # dont need to Normalize and Resize
    ])
    test_transform = Compose([
        ToTensor(),
    ])
    # forming datasets
    train_dataset = VOCDataset(root=args.data_path, year=args.year, image_set="train", download=False,
                               transform=train_transform)
    val_dataset = VOCDataset(root=args.data_path, year=args.year, image_set="val", download=False,
                             transform=test_transform)
    # loading datasets
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    # define the model
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                                              trainable_backbone_layers=0)
    # modify end layers
    in_channels = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_channels,
                                                      num_classes=len(train_dataset.categories))

    # optimizer the model
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9)
    best_map = -1
    start_epoch = 0
    # take model
    if args.saved_checkpoint:
        checkpoint = torch.load(args.saved_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["map"]
    else:
        start_epoch = 0
        best_map = -1

    # push model in gpu
    model.to(device)
    # make a path to save tensorboard
    if os.path.isdir(args.log_folder):
        shutil.rmtree(args.log_folder)
    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    # forming tensorboard
    writer = SummaryWriter(args.log_folder)
    num_iters_per_epoch = len(train_dataloader)
    # start training the model
    for epoch in range(start_epoch, args.num_epochs):
        # TRAINING PHASE
        model.train()
        train_loss = []
        # define progress_bar
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            # change images and labels into gpu
            images = [image.to(device) for image in images]
            labels = [{"boxes": target["boxes"].to(device), "labels": target["labels"].to(device)} for target in labels]
            # Forward
            losses = model(images, labels)
            final_losses = sum([loss for loss in losses.values()])
            # Backward
            # optimizer
            optimizer.zero_grad()
            final_losses.backward()
            optimizer.step()
            # write processing training on screen and into tensorboard
            train_loss.append(final_losses.item())
            mean_loss = np.mean(train_loss)
            progress_bar.set_description(
                "Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, args.num_epochs, mean_loss))
            writer.add_scalar("Train/Loss", mean_loss, epoch * num_iters_per_epoch + iter)

        # VALIDATION PHASE
        model.eval()
        test_progress_bar = tqdm(val_dataloader, colour="yellow")
        # define matrics to evaluate model
        metric = MeanAveragePrecision(iou_type="bbox")
        for iter, (images, labels) in enumerate(test_progress_bar):
            images = [image.to(device) for image in images]
            with torch.no_grad():
                outputs = model(images)
            # write results to matrics
            # change to cpu
            preds = []
            for output in outputs:
                preds.append({
                    "boxes": output["boxes"].to("cpu"),
                    "scores": output["scores"].to("cpu"),
                    "labels": output["labels"].to("cpu"),
                })

            targets = []
            for label in labels:
                targets.append({
                    "boxes": label["boxes"],
                    "labels": label["labels"],
                })
            metric.update(preds, targets)
        result = metric.compute()
        # write in tensorboard
        writer.add_scalar("Val/mAP", result["map"], epoch)
        writer.add_scalar("Val/mAP_50", result["map_50"], epoch)
        writer.add_scalar("Val/mAP_75", result["map_75"], epoch)
        # save model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "map": result["map"],
            "epoch": epoch + 1,
            "optimizer_state_dict": optimizer.state_dict()  # learning rate thay doi trong qua trinh huan luyen
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_folder, "last.pt"))
        if (result["map"] > best_map):
            best_map = result["map"]
            torch.save(checkpoint, os.path.join(args.checkpoint_folder, "best.pt"))


if __name__ == '__main__':
    args = get_args()
    train(args)
