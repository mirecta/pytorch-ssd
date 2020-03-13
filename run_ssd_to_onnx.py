from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import sys

import torch
from torch.autograd import Variable

print (sys.argv)
net_type = sys.argv[1]
model_path = sys.argv[2]

num_classes = int(sys.argv[3])


if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(num_classes, is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(num_classes, is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(num_classes, is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(num_classes, is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(num_classes, is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)


# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 3, 300, 300)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(net, dummy_input, sys.argv[4])
