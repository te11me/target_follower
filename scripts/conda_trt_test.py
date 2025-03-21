import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

print(f"[Success] TensorRT Version: {trt.__version__}")
print(f"CUDA Device: {cuda.Device(0).name()}")
