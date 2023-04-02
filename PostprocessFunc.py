import numpy as np
import pandas as pd
import os
import cv2
import random
import torch
from pathlib import Path
from tqdm import tqdm 
from Dataset import YOLODataset
from Model import Yolo
from collections import Counter
from Util import (
    intersection_over_union,
    non_max_suppression,
    cells_to_bboxes
)



os.chdir('./50_examples')

# some parameters to be specified
img_dir = os.path.join(os.getcwd(), "images")
label_dir = os.path.join(os.getcwd(), "labels")

img_files = sorted(os.listdir(img_dir))
label_files = sorted(os.listdir(label_dir))

annotation_path = os.path.join(os.getcwd(), "annotation.csv")

df = pd.DataFrame(data=[img_files, label_files]).T
df.to_csv(annotation_path, index=False)

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

image_size = 416
num_classes = 20
CONF_Thresh = 0.1
MAP_IoU_Thresh = 0.5
NMS_IoU_Thresh = 0.45
S = [13, 26, 52]  # [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
device = "cuda" if torch.cuda.is_available() else "cpu"



def check_class_accuracy(model, loader, threshold):
    
    """
    Parameter：
        loader: should be calculated by the function 'get_loader()', which can output
        training data loader or test data loader
    """
    
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to("cuda")
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to("cuda")
            obj = y[i][..., 0] == 1 # if there is an object
            noobj = y[i][..., 0] == 0  # no object detected

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            ) # check class predictions
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()





def get_evaluation_bboxes(loader, model, iou_threshold,
                          anchors, threshold,
                          box_format, device,
                         ):
    """
    This function can compute our prediction bboxes and true bboxes
    need functions: cells_to_bboxes() and non_max_suppression()
    make sure model is in eval before get bboxes
    Parameters:
        loader: loader of test data which should be evaluated
        box_format: "midpoint" or "corner"
    """
    
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold = iou_threshold,
                threshold = threshold,
                box_format = box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes




def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    This function calculates mean average precision (mAP).
    We may change the iouthreshold(maybe like 0.5,0.55,...,0.95) and store them,
    then take the average
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example
        # eg: amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # transfer to: ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections))) # true positive
        FP = torch.zeros((len(detections))) # false positive
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then can skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0) # cumulation
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)



""" 
This part can be deleted.

def test(pathname):
    imgList = os.listdir(pathname)
    #imgList.remove('.DS_Store')
    size = len(imgList)
    arr = np.zeros((size, 3, 416, 416))
    index = 0
    for filename in imgList:
        img = cv2.imread(pathname + '/' + filename)
        img = cv2.resize(img, (416, 416))
        B, G, R = cv2.split(img)
        img = np.array([R, G, B])
        arr[index] = img
        index += 1
    arr = torch.from_numpy(arr)
    arr = arr.float()
    num_classes = 20
    Model = Yolo(num_classes=num_classes)
    img_size = 416
    out = Model(arr)
    assert out[0].shape == (50, 3, img_size//32, img_size//32, 5 + num_classes)
    assert out[1].shape == (50, 3, img_size//16, img_size//16, 5 + num_classes)
    assert out[2].shape == (50, 3, img_size//8, img_size//8, 5 + num_classes)

pathname = '/Users/yiyi/Desktop/DSA5204/Project/Project-main/50_examples/images'
test(pathname)

dataset = YOLODataset(csv_file=annotation_path, img_dir=img_dir, label_dir=label_dir, anchors=ANCHORS)
model1 = Yolo(num_classes=num_classes)

def get_loader(test_dataset):
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=5,
        num_workers=5,
        pin_memory=4,
        shuffle=False,
        drop_last=False,
    )
    return test_loader

test_loader = get_loader(pathname + "annotation.csv")
"""

