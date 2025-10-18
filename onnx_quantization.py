# pip install onnxruntime onnxruntime-tools
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import numpy as np
from utils.misc import *
from utils.train_helpers import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='../imagenet-100/')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--input_name', default='input.1')
parser.add_argument('--rotation', default=True,type=bool)
parser.add_argument('--workers', default=8, type=int)


args = parser.parse_args()



# Implement CalibrationDataReader for your representative dataset
class MyCalibReader(CalibrationDataReader):
    def __init__(self, args):
        _, trloader = prepare_train_data(args)
        self.dataloader = iter(trloader)
        self.input_name = args.input_name
        self.counter = 0
    def get_next(self):
        try:
            if self.counter == 500:
                return None
            else:
                self.counter+=1
            arr = next(self.dataloader)[0].cpu().numpy()

            # Add batch dimension
            return {self.input_name: arr}

        except StopIteration:
            return None

calib_reader = MyCalibReader(args)

quantize_static(
    model_input='model.onnx',
    model_output='model_int8.onnx',
    calibration_data_reader=calib_reader,
    weight_type=QuantType.QInt8,      # weight quantized to int8 (per-channel)
    activation_type=QuantType.QInt8,  # activation int8 -> fully integer
)
