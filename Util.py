from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np

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

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()
