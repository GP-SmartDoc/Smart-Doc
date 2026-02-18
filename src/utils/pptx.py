from pptx import Presentation
from pptx.util import Inches
import json
import os
import re
from utils.strings import clean_json_string

def save_as_pptx(raw_llm_output, output_path="GP_Presentation.pptx"):
    #print(f"\n--- [RENDER] Starting PPTX Generation ---")
    prs = Presentation()
    
    cleaned_data = clean_json_string(raw_llm_output)
    
    try:
        slides_data = json.loads(cleaned_data)
        
        for i, data in enumerate(slides_data):
            # Layout 1 is 'Title and Content'
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            
            # 1. Set the Title
            slide.shapes.title.text = data.get("title", f"Slide {i+1}")
            
            # 2. Add Bullet Points (ensure we don't include paths here)
            body_shape = slide.placeholders[1]
            tf = body_shape.text_frame
            tf.clear() # Remove default text
            
            for point in data.get("content", []):
                p = tf.add_paragraph()
                p.text = str(point)
            
            # 3. FIX: Actually Insert the Image (don't just print the path)
            img_path = data.get("image_path")
            if img_path and os.path.exists(img_path):
                #print(f"[RENDER] Inserting image: {img_path}")
                # Position: Right side of the slide
                # (left, top, width, height)
                slide.shapes.add_picture(
                    img_path, 
                    left=Inches(6.5), 
                    top=Inches(1.5), 
                    width=Inches(3)
                )
            else:
                print(f"[WARNING] Image path invalid or missing: {img_path}")

        prs.save(output_path)
        print(f"--- [SUCCESS] Presentation saved to {output_path}")

    except Exception as e:
        print(f"--- [RENDER ERROR]: {e}")