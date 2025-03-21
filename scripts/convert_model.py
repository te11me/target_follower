#!/usr/bin/env python3
import torch
import torchvision
import onnx
import numpy as np
from onnxsim import simplify

def export_onnx_model():
    # 创建模型
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.eval()

    # 创建虚拟输入（固定batch_size=1）
    dummy_input = torch.randn(1, 3, 224, 224)  # 固定batch_size=1

    # 导出ONNX模型
    input_names = ["input"]
    output_names = ["output"]
    
    torch.onnx.export(
        model,
        dummy_input,
        "mobilenet_v2.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
        # 移除dynamic_axes参数
        # dynamic_axes={
        #     'input': {0: 'batch_size'},
        #     'output': {0: 'batch_size'}
        # }
    )

    # 优化模型时指定输入形状
    onnx_model = onnx.load("mobilenet_v2.onnx")
    simplified_model, check = simplify(
        onnx_model,
        input_shapes={'input': [1, 3, 224, 224]},  # 明确指定输入形状
        perform_optimization=True
    )
    onnx.save(simplified_model, "mobilenet_v2_sim.onnx")
    print(f"Simplified model check: {check}")

def verify_onnx_model():
    # 验证ONNX模型
    import onnxruntime
    ort_session = onnxruntime.InferenceSession("mobilenet_v2_sim.onnx")
    inputs = {"input": np.random.randn(1,3,224,224).astype(np.float32)}
    outputs = ort_session.run(None, inputs)
    print("ONNX model output shape:", outputs[0].shape)

if __name__ == "__main__":
    export_onnx_model()
    verify_onnx_model()
    print("Model conversion completed!")