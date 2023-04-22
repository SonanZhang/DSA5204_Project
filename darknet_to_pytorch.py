import torch
import torch.nn as nn
import sys
from Model import Yolo, CNNBlock, PredictionConvBlock, ScalePrediction, ResidualBlock
import numpy as np


def darknet_to_pytorch(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict


def load_darknet_weights(model, weights_path):
    with open(weights_path, 'rb') as f:
        # Skip the header
        _ = np.fromfile(f, dtype=np.int32, count=5)
        conv_ls = []
        for name, module_0 in model.named_modules():
            if isinstance(module_0, Yolo):
                for j in range(11):
                    module = module_0.layers[j]
                    if isinstance(module, CNNBlock) or isinstance(module, ResidualBlock):
                        if hasattr(module, 'num_repeats'):
                            for num_repeat in range(module.num_repeats):
                                conv_ls.append(module.layers[num_repeat][0].conv)
                                conv_ls.append(module.layers[num_repeat][1].conv)
                        else:
                            conv_ls.append(module.conv)

        for conv in conv_ls:
            bn = None
            if isinstance(conv, nn.Sequential):
                bn = conv[1]
                conv = conv[0]

            if bn is not None:
                # Load bias
                num_bias = bn.bias.numel()
                bn_bias = np.fromfile(f, dtype=np.float32, count=num_bias)
                if len(bn_bias) != 0:
                    bn_bias = torch.from_numpy(bn_bias).view_as(bn.bias)
                    bn.bias.data.copy_(bn_bias)

                # Load weight
                num_weight = bn.weight.numel()
                bn_weight = np.fromfile(f, dtype=np.float32, count=num_weight)
                if len(bn_weight) != 0:
                    bn_weight = torch.from_numpy(bn_weight).view_as(bn.weight)
                    bn.weight.data.copy_(bn_weight)

                # Load running mean
                num_running_mean = bn.running_mean.numel()
                bn_running_mean = np.fromfile(f, dtype=np.float32, count=num_running_mean)
                if len(bn_running_mean) != 0:
                    bn_running_mean = torch.from_numpy(bn_running_mean).view_as(bn.running_mean)
                    bn.running_mean.data.copy_(bn_running_mean)

                # Load running variance
                num_running_var = bn.running_var.numel()
                bn_running_var = np.fromfile(f, dtype=np.float32, count=num_running_var)
                if len(bn_running_var) != 0:
                    bn_running_var = torch.from_numpy(bn_running_var).view_as(bn.running_var)
                    bn.running_var.data.copy_(bn_running_var)

                # Load convolutional weights
                num_weights = conv.weight.numel()
                conv_weights = np.fromfile(f, dtype=np.float32, count=num_weights)
                if len(conv_weights) != 0:
                    conv_weights = torch.from_numpy(conv_weights).view_as(conv.weight)
                    conv.weight.data.copy_(conv_weights)
            elif conv.bias is not None:
                # Load bias
                num_bias = conv.bias.numel()
                conv_bias = np.fromfile(f, dtype=np.float32, count=num_bias)
                if len(conv_bias) != 0:
                    conv_bias = torch.from_numpy(conv_bias).view_as(conv.bias)
                    conv.bias.data.copy_(conv_bias)

                # Load convolutional weights
                num_weights = conv.weight.numel()
                conv_weights = np.fromfile(f, dtype=np.float32, count=num_weights)
                if len(conv_weights) != 0:
                    conv_weights = torch.from_numpy(conv_weights).view_as(conv.weight)
                    conv.weight.data.copy_(conv_weights)



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python darknet_to_pytorch.py <path_to_darknet_weights> <path_to_pytorch_weights>')
        sys.exit()

    input_darknet_weights = sys.argv[1]
    output_pytorch_weights = sys.argv[2]

    num_classes = 20  # For COCO dataset
    model = Yolo(num_classes=num_classes)
    load_darknet_weights(model, input_darknet_weights)

    # Save the state dictionary of the model
    torch.save(model.state_dict(), output_pytorch_weights)
    print(f'Successfully converted {input_darknet_weights} to {output_pytorch_weights}')
