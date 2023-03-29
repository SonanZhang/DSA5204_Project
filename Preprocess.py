import numpy as np
import os
import pandas as pd
import torch
import cv2

from matplotlib import pyplot as plt
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
# from utils import (
#     cells_to_bboxes, # only for testing
#     iou_width_height as iou,
#     non_max_suppression as nms, # only for testing
#     plot_image #only for testing
# )

ImageFile.LOAD_TRUNCATED_IMAGES = True


def anchor_iou (box_pred, box_anchor):
    w_1 = box_pred[..., 0:1]
    h_1 = box_pred[..., 1:2]
    w_2 = box_anchor[..., 0:1]
    h_2 = box_anchor[..., 1:2]

    w_max = torch.max(w_1, w_2)
    w_min = torch.min(w_1, w_2)
    h_max = torch.max(h_1, h_2)
    h_min = torch.min(h_1, h_2)


    return (w_min * h_min) / ( (w_1 * h_1) + (w_2 * h_2) - (w_min * h_min) )

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

"""
Creates a Pytorch dataset to load the Pascal VOC 
"""


class YOLODataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir,
            label_dir,
            anchors,
            image_size=416,
            S=[13, 26, 52],
            C=20,
            transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # apply augmentations with albumentations
        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # Building the targets below:
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = anchor_iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        return image, tuple(targets)


# Annotation file
os.chdir('/Users/lalala/Desktop/project-5204/Pascal VOC2012/train')

img_dir = os.path.join(os.getcwd(), "images")
label_dir = os.path.join(os.getcwd(), "labels")

img_files = sorted(os.listdir(img_dir))
label_files = sorted(os.listdir(label_dir))

annotation_path = os.path.join(os.getcwd(), "annotation.csv")

df = pd.DataFrame(data=[img_files, label_files]).T
df.to_csv(annotation_path, index=False)


# Call Class
def draw_boxes(image, boxes, class_labels):
    """
    Draws bounding boxes on an image given the predicted boxes and class labels.

    Arguments:
    image -- NumPy array of shape (H, W, C) representing the input image.
    boxes -- NumPy array of shape (N, 4) representing the predicted bounding boxes in the format (x, y, w, h).
    class_labels -- NumPy array of shape (N,) representing the predicted class labels.

    Returns:
    output_image -- NumPy array of shape (H, W, C) representing the input image with bounding boxes drawn.
    """
    output_image = np.copy(image)
    for i, box in enumerate(boxes):
        x, y, w, h = box
        xmin, ymin, xmax, ymax = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        class_label = class_labels[i]
        color = (0, 255, 0)  # green
        thickness = 1
        output_image = cv2.rectangle(output_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255),
                                     thickness)
        output_image = cv2.putText(output_image, class_label, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                   color, thickness, cv2.LINE_AA)

    return output_image


def visualize_output(image, targets, anchors):
    """
    Visualizes the predicted bounding boxes and class labels on the input image.

    Arguments:
    image -- NumPy array of shape (H, W, C) representing the input image.
    targets -- Tuple of PyTorch tensors representing the target outputs from the YOLO model.
    anchors -- List of anchor boxes used by the YOLO model.

    Returns:
    None
    """
    # Define a default set of class names
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']

    # Get the predicted bounding boxes and class labels
    boxes = []
    class_labels = []
    for i, target in enumerate(targets):
        S = target.shape[1]
        num_anchors = len(anchors[i])
        for row in range(S):
            for col in range(S):
                for n in range(num_anchors):
                    if target[n, row, col, 0] > 0.5:
                        x_cell, y_cell, w_cell, h_cell = target[n, row, col, 1:5].tolist()
                        x = (col + x_cell) * (image.shape[1] / S)
                        y = (row + y_cell) * (image.shape[0] / S)
                        w = w_cell * anchors[i][n][0]
                        h = h_cell * anchors[i][n][1]
                        class_label = int(target[n, row, col, 5])
                        boxes.append([x, y, w, h])
                        class_labels.append(class_names[class_label])

    # Draw the boxes on the input image
    output_image = draw_boxes(image, boxes, class_labels)

    # Display the output image
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

dataset = YOLODataset(csv_file=annotation_path, img_dir=img_dir, label_dir=label_dir, anchors=ANCHORS)

image, targets = dataset[56]

visualize_output(image, targets, ANCHORS)