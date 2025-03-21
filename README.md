# target_follower
目标跟踪和特征识别的部分。获取图像到最终模型识别并跟踪目标。  深度相机和激光雷达的数据融合。使用CNN进行特征提取，用预训练的MobileNet。图像中提取特征，目标跟踪。 LKCF算法的部分融合深度和RGB数据。LKCF进行跟踪。模型转换从ONNX模型生成TensorRT引擎。  目标消失后重新特征匹配和重识别
环境:
jetson TX1,ubuntu18.04,melodic(opencv3.2.0+opencv4.1.1),独立文件夹源码编译安装opencv4.5.5
