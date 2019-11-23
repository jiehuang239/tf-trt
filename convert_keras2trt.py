import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
input_saved_model_dir = 'resnet_model/'
output_saved_model_dir = 'resnet50_saved_model_TFTRT_FP16'
converter = trt.TrtGraphConverter(input_saved_model_dir=input_saved_model_dir,max_workspace_size_bytes=16000000000,precision_mode=trt.TrtPrecisionMode.FP16)
converter.convert()
converter.save(output_saved_model_dir)
