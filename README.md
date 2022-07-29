# YOLOv7 Inference using OpenVINO runtime
Yolov7 model inference demo
## 1 set OpenVINO env
```shell
  python3 -m venv openvino_env
  source openvino_env/bin/activate
  python -m pip install --upgrade pip
  pip install openvino==2022.1.0
  python -c "from openvino.runtime import Core"
 ```
## 2 Download  pre-trained model
```shell
     from (https://github.com/WongKinYiu/yolov7)
 ```
## 3 Convert the pth model to ONNX model
```shell
   python3 export.py --weights yolov7.pt
 ```
 
## 4 Run inference
 ```shell
  1)put the model in model dir and test image in data
  2)run: python3 main.py
 ```
