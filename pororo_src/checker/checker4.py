# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image, ImageFile
from pororo import Pororo
import json
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import gc  # 가비지 컬렉터 임포트
import torch
import datetime
import re
import Levenshtein

warnings.filterwarnings('ignore')  # 경고 메시지 무시
ImageFile.LOAD_TRUNCATED_IMAGES = True
start_time = datetime.datetime.now()


# OCR 모델 초기화
ocr = Pororo(task="ocr", lang="ko", model="brainocr")
# IoU를 계산하는 함수 정의
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 두 박스의 좌상단 점과 우하단 점 좌표 계산
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    # 겹치는 영역 계산
    x_intersection = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_intersection = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    # IoU 계산
    intersection_area = x_intersection * y_intersection
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou

#list 사이 intersection부분
def Intersection(lst1, lst2):
    return set(lst1).intersection(lst2)

def is_image_corrupted(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 이미지 파일 검증
            del img
        return False  # 검증이 성공하면 파일은 손상되지 않음
    except (IOError, SyntaxError) as e:
        print(f"손상된 이미지 파일: {file_path}, 오류: {e}")
        del img
        return True  # 검증 중 오류 발생 시 파일은 손상됨

# 사용자로부터 폴더 경로 입력 받기
folder_path = "./"

# CSV 파일 불러오기 또는 초기화
autosave_file = "auto_validation_autosave2.csv"
if os.path.exists(autosave_file):
    result_df = pd.read_csv(autosave_file, encoding="utf-8")
    processed_files_set = set(result_df['json_file'])  # 이미 처리된 파일 집합
else:
    columns = ["json_file", "image_file", "ocr_list", "ocr_result", "ocr_count", "ocr_result_fail_count", "boundingbox_result", "table_type_result","description_result"]
    result_df = pd.DataFrame(columns=columns)
    processed_files_set = set()

# 폴더 내 총 .json 파일 개수 카운트 (중복 제거)
total_files = sum(
    1
    for root, _, files in os.walk(os.path.join(folder_path, "Labeling"), topdown=True)
    for file in files
    if file.endswith(".json") and file not in processed_files_set
)
processed_files = 0;

# 폴더 내 모든 .json 파일을 찾아서 OCR 수행
for root, dirs, files in os.walk(os.path.join(folder_path, "Labeling"), topdown=True):
    for file in files:
        if file.endswith(".json") and file not in processed_files_set:
            print('ddd',file)
            processed_files += 1  # 처리된 파일 수 증가
            json_file_path = os.path.join(root, file)
            
            # Search for the corresponding image file in the "image" folder and its subdirectories
            image_folder_path = os.path.join(folder_path, "image")
            image_file_path=''
            for image_root, _, image_files in os.walk(image_folder_path):
                for image_file in image_files:
                    print('immagg',image_file)
                    if image_file.endswith(".png") and image_file.startswith(file[:-5]):  # Match the image file by name
                        image_file_path = os.path.join(image_root, image_file)
                        break

            # 이미지 파일 열기
            try:
                image = Image.open(image_file_path)
                image_resize_ratio = 2 if max(image.width, image.height)/2 <= 960 else 4
                image_resized = image.resize((image.width // image_resize_ratio, image.height // image_resize_ratio))  # 이미지 크기를 50%로 줄이기

                print(f"처리 중인 파일: {file} | 완료된 갯수 : {processed_files} / 총 갯수 : {total_files} | Resize Ratio : {image_resize_ratio}")  # 파일명과 진행 상태 출력

                ocr_image = ocr(np.array(image_resized))
            except Exception as e:
                print(f"이미지 파일을 찾을 수 없습니다: {image_file_path} Error:{e}")
                ocr_image = []
                continue

            try:
                # 라벨링 데이터 파일 열기
                with open(json_file_path, "r", encoding="utf-8") as label_file:
                    label_data = json.load(label_file)
            except Exception as e:
                print(f"JSON 파일 파싱 오류: {label_file}")
                continue

            first_iteration = True  # 첫 번째 반복을 확인하기 위한 플래그

            bbox_padding = 1 #바운딩박스에 여백을 두어 글자인식이 잘 안된것을 개선

            # Bounding Box 영역 검수
            for label_info in label_data["annotations"]["ocr_labels"]:
                if label_info["type"] == "bbox":
                    bbox = label_info["bbox"]
                    x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

                    try:
                        cropped_image = np.array(image.crop((x-bbox_padding, y-bbox_padding, x + width+bbox_padding, y + height+bbox_padding)))  # Bounding Box 영역 자르기
                        ocr_result = ocr(cropped_image,detail=True)  # Cropped Image 에 대한 OCR 수행
                    except Exception as e:
                        print(f"{file}의 Bounding Box 영역 OCR 처리 중 오류 발생: {e}")
                        continue
                    
                    #BOUNDINGBOX IOU
                    iou = 0 #bounding_poly 없는 경우 iou=0
                    if len(ocr_result["bounding_poly"])>0:
                        ocr_bbox_x ,ocr_bbox_y = ocr_result["bounding_poly"][0]["vertices"][0]["x"], ocr_result["bounding_poly"][0]["vertices"][0]["y"] #좌상단
                        ocr_bbox_rd = ocr_result["bounding_poly"][0]["vertices"][2] #우하단
                        ocr_bbox_w, ocr_bbox_h = ocr_bbox_rd["x"] - ocr_bbox_x, ocr_bbox_rd["y"] - ocr_bbox_y
                        iou = calculate_iou([x,y,width,height],[ocr_bbox_x+x-bbox_padding, ocr_bbox_y+y-bbox_padding, ocr_bbox_w, ocr_bbox_h])
                    
                    # OCR 결과와 기대값 비교
                    expected_text = label_info["text"]
                    ocr_result_str = ' '.join(ocr_result)  # Join the list of words into a single string

                    # ... [음절 단위 비교 및 오류 카운트 코드]
                    # 음절(글자) 단위로 분리
                    ocr_result_chars = ''.join(ocr_result["description"])
                    expected_text_chars = expected_text

                    #특수문자 제거, 한글영어숫자를 제외한 일어 중국어 등 모든 문자 제거
                    pattern = r'[^\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318Fa-zA-Z0-9\s]'
                    cet = re.sub(pattern,'',expected_text_chars)
                    corc = re.sub(pattern,'',ocr_result_chars)
                    cleaned_expected_text = list(filter(None, cet.split(' '))) # remove empty stirings
                    cleaned_ocr_result_chars = list(filter(None, corc.split(' '))) # remove empty stirings
                    # print('특수문자 일본어 중국어 등 제외된 기대값:',cleaned_expected_text)
                    # print('특수문자 일본어 중국어 등 제외된 ocr값:',cleaned_ocr_result_chars)

                    # 어절 단위로 두 문장 사이 틀린 어절수
                    ocr_fail_count = len(cleaned_expected_text) - len(Intersection(cleaned_expected_text, cleaned_ocr_result_chars))

                    #ocr_result는 오타1개까지 허용 
                    expected_text_rest = [item for item in cleaned_expected_text if item not in cleaned_ocr_result_chars]
                    ocr_result_rest = [item for item in cleaned_ocr_result_chars if item not in cleaned_expected_text]
                    str_expected =''.join(expected_text_rest)
                    str_ocr =''.join(ocr_result_rest)
                    # print('동일값 제외한 나머지 expected: ',str_expected)
                    # print('동일값 제외한 나머지 ocr: ',str_ocr)
                    # print('levenshtein distance : ',Levenshtein.distance(str_expected,str_ocr))
                    l_distance = Levenshtein.distance(str_expected,str_ocr)
                    
                    # 예외 > 일본어 중국어 한자 ? fail -->  fail 어절수를 1로 기록 
                    if cet == '':
                        ocr_fail_count = 1
                        # print(expected_text_chars)
                        # print(ocr_result_chars)
                        # l_distance = Levenshtein.distance(expected_text_chars,ocr_result_chars) #특수문자 오타수
                        # print(Levenshtein.distance(expected_text_chars,ocr_result_chars))

                    #다음은 ~에 대한 이미지이고, ~에 대한 정보를 담고 있습니다 iod 정확도를 위해 제거
                    cdstr = re.sub(r'^다음은|에 대한 이미지이고|에 대한|정보가 포함되어 있습니다|정보를 담고 있습니다',' ',label_data["descriptions"]["description"])
                    cdstr2 = re.sub(pattern,'',cdstr) #영어,한글,숫자 제외한 중국어 일본어 특수문자 모두 제외
                    clenaed_description = re.sub(r'\n',' ',cdstr2) #개행문자는 띄어쓰기로 변환
                    cleaned_auto_entire_ocr_texts = re.sub(pattern,'',', '.join(ocr_image))
                    description_lst = list(filter(None, clenaed_description.split(' '))) # remove empty stirings
                    auto_entire_lst = list(filter(None, cleaned_auto_entire_ocr_texts.split(' '))) # remove empty stirings

                    #라벨링된 description <-> OCR 라벨링 여기서 추출한 글자가 50프로 이상 포함되어있으면 Pass or Fail
                    iod = len(Intersection(description_lst,auto_entire_lst)) / len(description_lst)
                    description_r = "PASS" if iod>0.5 else "FAIL"

                    new_line = '\n'
                    # 결과를 데이터프레임에 추가
                    result_df = result_df._append({
                        "json_file": file,
                        "image_file": os.path.basename(image_file_path),
                        "ocr_list": expected_text,  # "ocr_list"를 기대값으로 설정
                        "ocr_result": "PASS" if l_distance < 2 else "FAIL", #오타 1개까지는 패스
                        "ocr_count": len(expected_text.split(' ')),
                        "ocr_result_fail_count": ocr_fail_count,
                        "boundingbox_result": "PASS" if iou > 0.8 else "FAIL",
                        "table_type_result": "PASS",
                        "description_result": description_r if first_iteration else "",
                        "auto_cropped_ocr_texts": f'ocr 추출 결과:{ocr_result["description"]} / 특수문자 및 중국어,일본어 등 제외 오타 수: {l_distance}개',
                        "auto_entire_ocr_texts": ', '.join(ocr_image) if first_iteration else "",
                    }, ignore_index=True)

                    first_iteration = False

                    del cropped_image, ocr_result, expected_text, ocr_result_str
                    gc.collect()
                    torch.cuda.empty_cache()
            
            print(datetime.datetime.now() - start_time)
            if processed_files % 10 == 0:
                result_df.to_csv(autosave_file, index=False, encoding="utf-8", errors='ignore')
                print(f"중간 저장 완료: {autosave_file}")

            del image, image_resized, ocr_image, label_data
            gc.collect()
            torch.cuda.empty_cache()

# 마지막 결과 저장
result_df.to_csv("auto_validation_final2.csv", index=False, encoding="utf-8", errors='ignore')
print("최종 검수 결과 저장 완료: auto_validation_final2.csv")

end_time = datetime.datetime.now()

print(end_time - start_time)
# 결과 출력
print("검수가 완료되었습니다.")
