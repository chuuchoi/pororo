from main import PororoOcr

ocr = PororoOcr()
print(ocr.get_available_models())
img_path='image/test00.png'
ocr.run_ocr(img_path, debug="True")