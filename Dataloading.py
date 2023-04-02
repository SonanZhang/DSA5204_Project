from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

DATASET = '/Users/zhangjianan/Desktop/Project/Pascal_VOC_2012'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
TRAIN_IMG_DIR = DATASET + "/train/images/"
TRAIN_LABEL_DIR = DATASET + "/train/labels/"
VALID_IMG_DIR = DATASET + "/valid/images/"
VALID_LABEL_DIR = DATASET + "/valid/labels/"
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=15, p=0.5, mode="constant"),
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)

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


def get_loaders(train_csv_path, test_csv_path):
    from Dataset import YOLODataset

    IMAGE_SIZE = 416
    train_dataset = YOLODataset(
        train_csv_path,
        transform=train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=TRAIN_IMG_DIR,
        label_dir=TRAIN_LABEL_DIR,
        anchors=ANCHORS,
    )
    test_dataset = YOLODataset(
        test_csv_path,
        transform=test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=VALID_IMG_DIR,
        label_dir=VALID_LABEL_DIR,
        anchors=ANCHORS,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    train_eval_dataset = YOLODataset(
        train_csv_path,
        transform=test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=VALID_IMG_DIR,
        label_dir=VALID_LABEL_DIR,
        anchors=ANCHORS,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_eval_loader