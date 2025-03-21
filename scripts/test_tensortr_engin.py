import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_path):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("mobilenet_v2.engine")
if engine:
    print("[Success] Engine loaded. Number of bindings:", engine.num_bindings)
    for i in range(engine.num_bindings):
        print(f"Binding {i}:", engine.get_binding_name(i), engine.get_binding_shape(i))
else:
    print("[Error] Failed to load engine")
