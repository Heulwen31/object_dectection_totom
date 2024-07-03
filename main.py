"""Module providing a function printing python version."""
import cv2
import torch
from torchvision.transforms import functional as f

from sound import text_to_mp3

# Thiết lập camera
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có mở không
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Tải mô hình Fast R-CNN đã huấn luyện
# model = fasterrcnn_resnet50_fpn(pretrained=True)

model = torch.load('/home/namtd/Desktop/totom_object_detection/model/entire_model_v2.pth', map_location=torch.device('cpu'))
model.eval()

# Thiết lập tên các nhãn (labels) COCO
COCO_INSTANCE_CATEGORY_NAMES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light'
]


# # Hàm để chuyển đổi ảnh từ định dạng OpenCV sang Tensor
def transform_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = f.to_tensor(image)
    return image


# Hàm để hiển thị kết quả
def display_result(image, output):
    boxes = output[0]['boxes'].cpu().detach().numpy().astype(int)
    labels = output[0]['labels'].cpu().detach().numpy()
    scores = output[0]['scores'].cpu().detach().numpy()

    for i in range(len(boxes)):
        if scores[i] > 0.8:
            x1, y1, x2, y2 = boxes[i]
            label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Result', image)


text_sound = 'xin chào'

while True:
    # Đọc một frame từ camera
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #     # Chuyển đổi ảnh sang tensor
    image_tensor = transform_image(frame)
    #     # Phát hiện đối tượng
    with torch.no_grad():
        outputs = model([image_tensor])
    print(outputs)

    if len(outputs[0]['scores']) > 0:
        threshold = outputs[0]['scores'].max()
        filtered_prediction = {
            'boxes': outputs[0]['boxes'][outputs[0]['scores'] >= threshold],
            'labels': outputs[0]['labels'][outputs[0]['scores'] >= threshold],
            'scores': outputs[0]['scores'][outputs[0]['scores'] >= threshold],
            'masks': outputs[0]['masks'][outputs[0]['scores'] >= threshold]
        }

        text_sound = COCO_INSTANCE_CATEGORY_NAMES[filtered_prediction['labels'][0]]

        outputs = [filtered_prediction]
    else:
        text_sound = "background"

    # Hiển thị kết quả
    display_result(frame, outputs)
    key = cv2.waitKey(1)

    if key == ord('q'):
        # Nhấn phím 'q' để thoát chương trình
        print("Quit")
        break
    elif key == ord('s'):
        # Nhấn phím 's' để lưu khung hình
        # cv2.imwrite('snapshot.png', frame)
        print("Sound process")
        text_to_mp3(text_sound, "sound")

# # Giải phóng camera và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
