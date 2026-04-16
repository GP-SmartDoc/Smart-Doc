import easyocr
import pathlib
import os
import time

print("TESTING easyocr\n")
# 1. Initialize the reader (specify the language, e.g., 'en' for English)
# Note: The first time you run this, it will download the necessary AI models.
model_path = pathlib.Path(__file__).parent.parent.parent / "models"
reader = easyocr.Reader(['en'], model_storage_directory=model_path, gpu=True)

# 2. Extract the text
# detail=0 returns just the text. detail=1 returns bounding boxes and confidence scores.
start = time.perf_counter()
results = reader.readtext(os.path.abspath('basic_image.png'), detail=0)
end = time.perf_counter()
print(f"Extracted Text (in {end-start:.6f} seconds):")
for line in results:
    print(line)
print("\n\n")

results = reader.readtext('blurred_image.png', detail=0)
print("Extracted Text:")
for line in results:
    print(line)
print("\n\n")

results = reader.readtext('flipped_image.png', detail=0)
print("Extracted Text:")
for line in results:
    print(line)
    
# print("\nTESTING paddleocr")

# from paddleocr import PaddleOCR

# ocr = ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
#     enable_mkldnn=False
# )
# print("Basic:")
# result = ocr.predict(input='basic_image.png')
# for res in result:
#     res.print()
# print("\n")

# print("Blurred:")
# result = ocr.predict(input='blurred_image.png')
# for res in result:
#     res.print()
# print("\n")

# print("Flipped:")
# result = ocr.predict(input='flipped_image.png')
# for res in result:
#     res.print()
# print("\n")

