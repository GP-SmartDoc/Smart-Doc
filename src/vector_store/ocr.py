import easyocr
import pathlib


model_path = pathlib.Path(__file__).parent.parent / "models"
reader = easyocr.Reader(['en'], model_storage_directory=model_path, gpu=True)

def perform_ocr(image_path:str)->str:
    results = reader.readtext(image_path, detail=0)
    ret = ""
    for line in results:
        ret += line
    return ret