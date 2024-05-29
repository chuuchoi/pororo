import os
import numpy as np
from PIL import Image
from pororo import Pororo
import json
import matplotlib.pyplot as plt

# OCR 모델 초기화
ocr = Pororo(task="ocr", lang="ko", model="brainocr")

# 사용자로부터 폴더 경로 입력 받기
folder_path = input("폴더 경로를 입력하세요: ")

# 폴더 내 모든 .json 파일을 찾아서 OCR 수행
for root, dirs, files in os.walk(os.path.join(folder_path, "labeling")):
    for file in files:
        if file.endswith(".json"):
            json_file_path = os.path.join(root, file)
            image_file_path = os.path.join(folder_path, "image", file.replace(".json", ".png"))

            # 이미지 파일 열기
            try:
                image = Image.open(image_file_path)
            except FileNotFoundError:
                print(f"이미지 파일을 찾을 수 없습니다: {image_file_path}")
                continue

            # 라벨링 데이터 파일 열기
            with open(json_file_path, "r", encoding="utf-8") as label_file:
                label_data = json.load(label_file)

            # OCR 수행
            for label_info in label_data["ocr_labels"]:
                if label_info["type"] == "bbox":
                    bbox = label_info["bbox"]
                    x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                    
                    # Bounding Box 영역 자르기
                    cropped_image = np.array(image.crop((x, y, x + width, y + height)))  # PIL 이미지를 numpy 배열로 변환
                    
                    # 자른 바운딩 박스를 Figure로 표시
                    plt.imshow(cropped_image)
                    plt.title(f"Label Class: {label_info['_class']}")
                    plt.show()

                    # OCR 수행
                    ocr_result = ocr(cropped_image)
                    
                    # OCR 결과 출력
                    print(f"Label Class: {label_info['_class']}")
                    print(f"Text: {ocr_result}")
                    print("=" * 50)
