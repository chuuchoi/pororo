import cv2
from pororo import Pororo
from pororo.pororo import SUPPORTED_TASKS
from utils.image_util import plt_imshow, put_text  # 이미지 유틸리티 함수 임포트
import warnings

warnings.filterwarnings('ignore')  # 경고 메시지 무시

# PororoOcr 클래스 정의
class PororoOcr:
    def __init__(self, model: str = "brainocr", lang: str = "ko", **kwargs):
        self.model = model
        self.lang = lang
        self._ocr = Pororo(task="ocr", lang=lang, model=model, **kwargs)  # Pororo OCR 모델 초기화
        self.img_path = None
        self.ocr_result = {}

    # OCR 수행 메서드
    def run_ocr(self, img_path: str, debug: bool = False):
        self.img_path = img_path
        self.ocr_result = self._ocr(img_path, detail=True)  # 이미지에서 텍스트 추출

        if self.ocr_result['description']:
            ocr_text = self.ocr_result["description"]
        else:
            ocr_text = "텍스트를 감지할 수 없습니다."

        if debug:
            self.show_img_with_ocr()  # 디버그 모드에서 이미지와 OCR 결과를 함께 표시

        return ocr_text

    # 사용 가능한 언어 목록 반환
    @staticmethod
    def get_available_langs():
        return SUPPORTED_TASKS["ocr"].get_available_langs()

    # 사용 가능한 OCR 모델 목록 반환
    @staticmethod
    def get_available_models():
        return SUPPORTED_TASKS["ocr"].get_available_models()

    # OCR 결과 반환
    def get_ocr_result(self):
        return self.ocr_result

    # 이미지 경로 반환
    def get_img_path(self):
        return self.img_path

    # 이미지 표시
    def show_img(self):
        plt_imshow(img=self.img_path)

    # OCR 결과가 표시된 이미지 표시
    def show_img_with_ocr(self):
        img = cv2.imread(self.img_path)
        roi_img = img.copy()

        for text_result in self.ocr_result['bounding_poly']:
            text = text_result['description']
            tlX = text_result['vertices'][0]['x']
            tlY = text_result['vertices'][0]['y']
            trX = text_result['vertices'][1]['x']
            trY = text_result['vertices'][1]['y']
            brX = text_result['vertices'][2]['x']
            brY = text_result['vertices'][2]['y']
            blX = text_result['vertices'][3]['x']
            blY = text_result['vertices'][3]['y']

            pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))

            topLeft = pts[0]
            topRight = pts[1]
            bottomRight = pts[2]
            bottomLeft = pts[3]

            cv2.line(roi_img, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(roi_img, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(roi_img, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(roi_img, bottomLeft, topLeft, (0, 255, 0), 2)
            roi_img = put_text(roi_img, text, topLeft[0], topLeft[1] - 20, font_size=15)

        plt_imshow(["원본 이미지", "ROI 이미지"], [img, roi_img], figsize=(16, 10))

if __name__ == "__main__":
    ocr = PororoOcr()
    image_path = input("이미지 경로를 입력하세요: ")
    text = ocr.run_ocr(image_path, debug=True)
    print('결과 :', text)