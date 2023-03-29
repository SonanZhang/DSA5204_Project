from matplotlib import pyplot as plt
import torch
import cv2

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
