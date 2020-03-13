#!/usr/bin/python3

from torch.autograd import Variable
import sys

# Load the trained model from file
trained_model = Net()
trained_model.load_state_dict(torch.load(sys.argv[1]))

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 1, 28, 28)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(trained_model, dummy_input, sys.argv[2])
