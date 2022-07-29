
from  detector import Yolov7
import cv2

if __name__ == "__main__":
    modelpath = './model/yolov7.onnx'
    img_path = './data/1.jpg'
    img = cv2.imread(img_path)

    detector = Yolov7(modelpath)
    result=detector.inference(img)

    for obj in result:
        cv2.rectangle(img, (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])), (213, 32, 255), cv2.LINE_AA)
        cv2.putText(img, 'person', (int(obj[0]), int(obj[1]) - 4), 0, 3, [225, 0, 255], thickness=3, lineType=cv2.LINE_AA)
    cv2.imwrite("./data/res1.jpg", img)

    print('done')