from pptx import Presentation
import json
import os
import re
import copy
from lxml import etree

# Semantic Map to help the LLM select the right template slides
LAYOUT_MAP = {
    "title_subtitle": 0, "title_content": 1, "three_column": 2,
    "content_image": 3, "large_image": 4, "title_only": 5
}

# Namespaces for PowerPoint XML manipulation
NS = {
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
}

def clean_json_string(raw_str):
    if not raw_str: return ""
    # Matches the outermost JSON array to ignore conversational filler
    match = re.search(r'\[\s*\{.*\}\s*\]', raw_str, re.DOTALL)
    return match.group(0) if match else ""

def format_value(val):
    """
    Handles nested LLM structures. Converts lists or dictionaries 
    into clean, bulleted strings for the slide.
    """
    if isinstance(val, list):
        return "\n".join([f"• {item}" for item in val])
    if isinstance(val, dict):
        # Recursively format dicts (handles nested 'title' and 'content' in columns)
        parts = []
        for k, v in val.items():
            if k.lower() == 'title':
                parts.append(str(v).upper())
            else:
                parts.append(format_value(v))
        return "\n".join(parts)
    return str(val)

def get_deep_value(data, key_to_find):
    """Recursively search for keys even if nested."""
    if key_to_find in data: return data[key_to_find]
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict):
                res = get_deep_value(v, key_to_find)
                if res: return res
    return None

def duplicate_slide(prs, source_index):
    """Exact deep copy of a slide preserving animations and layout."""
    source_slide = prs.slides[source_index]
    dest_slide = prs.slides.add_slide(source_slide.slide_layout)

    # 1. Clear default placeholders but keep structure
    for shp in list(dest_slide.shapes):
        dest_slide.shapes._spTree.remove(shp._element)

    # 2. Copy shapes preserving relationships and IDs
    for element in source_slide.shapes._spTree:
        new_element = copy.deepcopy(element)
        # Unique ID generation to prevent corruption
        if 'id' in new_element.attrib:
            new_id = str(hash(new_element))[-5:]
            new_element.attrib['id'] = new_id
        dest_slide.shapes._spTree.append(new_element)

    # 3. Copy transitions and timing (Animations)
    slide_elem = source_slide._element
    for tag in ['transition', 'timing']:
        node = slide_elem.find(f'./p:{tag}', namespaces=NS)
        if node is not None:
            # Append in correct XML order
            dest_slide._element.append(copy.deepcopy(node))

    # 4. Copy background
    csld = slide_elem.find('./p:cSld', namespaces=NS)
    if csld is not None:
        bg = csld.find('./p:bg', namespaces=NS)
        if bg is not None:
            dest_csld = dest_slide._element.find('./p:cSld', namespaces=NS)
            if dest_csld is not None:
                dest_csld.insert(0, copy.deepcopy(bg))

    return dest_slide

def save_as_pptx(raw_llm_output, template_path="layouts.pptx", output_path="generated_slides.pptx"):
    if not os.path.exists(template_path): return
    
    prs = Presentation(template_path)
    template_count = len(prs.slides)
    cleaned_data = clean_json_string(raw_llm_output)
    
    try:
        slides_data = json.loads(cleaned_data)
        for data in slides_data:
            # FIX 1: Robust layout mapping (check 'layout' and 'layout_name')
            layout_key = data.get("layout_name", data.get("layout", "title_content"))
            if layout_key == "content": 
                layout_key = "content_image" if data.get("image_path") or data.get("image") else "title_content"
            
            idx = LAYOUT_MAP.get(layout_key, 1)
            new_slide = duplicate_slide(prs, idx)

            # FIX 2: Better data mapping for tags
            # Merges root data with 'tags' dict if it exists
            content_source = data.get("tags", {})
            content_source.update({k: v for k, v in data.items() if k not in ["tags", "layout", "layout_name", "image", "image_path"]})

            # --- TAG REPLACEMENT ---
            for shape in new_slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            for key in ["title", "subtitle", "content", "col1", "col2", "col3"]:
                                val = content_source.get(key)
                                if val:
                                    # Uses format_value to handle lists and objects (e.g. nested three_column data)
                                    formatted_text = format_value(val)
                                    tag = "{{" + key + "}}"
                                    if tag in run.text:
                                        run.text = run.text.replace(tag, formatted_text)
                            
                            # Cleanup empty tags leftover by the LLM
                            if "{{" in run.text and "}}" in run.text:
                                run.text = re.sub(r'\{\{.*?\}\}', '', run.text)
            
            # --- IMAGE REPLACEMENT (Fix 3: Handle 'image' vs 'image_path') ---
            img_path = data.get("image_path", data.get("image"))
            if img_path and isinstance(img_path, str) and img_path.strip():
                # Check for Image in local directory if path is broken
                if not os.path.exists(img_path):
                    base_name = os.path.basename(img_path)
                    img_path = os.path.join("Images", base_name) if os.path.exists(os.path.join("Images", base_name)) else img_path

                if os.path.exists(img_path):
                    for shape in list(new_slide.shapes):
                        if "DUMMY" in (shape.name or "").upper():
                            left, top, width, height = shape.left, shape.top, shape.width, shape.height
                            new_slide.shapes._spTree.remove(shape._element)
                            new_slide.shapes.add_picture(img_path, left, top, width, height)
                            break

        # Safe Cleanup of template slides
        for i in range(template_count - 1, -1, -1):
            slide_elem = prs.slides._sldIdLst[i]
            prs.part.drop_rel(slide_elem.rId)
            prs.slides._sldIdLst.remove(slide_elem)

        prs.save(output_path)
        print(f"Success: Repaired file with animations saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")