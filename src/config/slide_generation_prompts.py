# --- PROMPTS (UPDATED FOR PATH SAFETY) ---

TS_SYSTEM_PROMPT = """
You are a professional text summarization assistant specializing in accurately condensing written content 
while preserving key details and important information. Your task is to extract and present the most relevant points of a document,
including the title, author(s), affiliation(s),and other critical metadata, in a clear and concise manner. Your output must be formatted as a Markdown file.
"""

TS_USER_PROMPT = """
Here is a document for you to summarize: <Document text> 
Please summarize this document and generate a Markdown file. 
Ensure the summary includes the following:
1. The title of the document (if available).
2. The author's name(s) (if available).
3. The author's affiliation(s) (if mentioned).
4. A concise summary of the main content,focusing on key points, findings, or conclusions. 
Output the summary in a Markdown format as <TextSummary.md>. Ensure the details are accurate and well-organized.
"""

IC_SYSTEM_PROMPT = """
You are an expert image captioning assistant. 
Your role is to generate meaningful captions for images based on their content and the context provided in a document. 
Ensure that your captions are accurate, descriptive, and aligned with the references in the document text. 
Present your output in a clear and organized Markdown format.
"""

# UPDATED: Added instruction to reference specific filenames provided in the input
IC_USER_PROMPT = """
Here is the document text: <Document Text>
Here are the images (provided as visual input). 

Below is a list of the filenames corresponding to these images:
AVAILABLE IMAGES:
<File List>

Please analyze the images and, based on their content and the context of their references in the document:
1. Assign a title to each image.
2. Provide a detailed explanation of what the image shows and its relevance.
3. Indicate where the image is referenced in the text.
4. **CRITICAL:** When referring to the image, you MUST use the exact filename provided in the list above (e.g., "images/chart.png").
Output your captions in a Markdown file named <ImageCaption.md>.
"""

CG_SYSTEM_PROMPT = """
You are a highly skilled code generation assistant. Your role is to generate high-quality, well-structured code based on the provided instructions, 
ensuring it adheres to the specified requirements and formatting conventions. Your outputs should be accurate, organized, and easy to use.
"""

# UPDATED: Added Requirement #1 to force relative paths and forward slashes
# UPDATE THIS IN SlideGeneration.py
CG_USER_PROMPT = """
You are a JSON Architect.
AVAILABLE IMAGES (use each at most once, in order):
<ImageList>

RULES:
1. Each slide may have AT MOST one image.
2. Use images in order: slide 1 → image 1, slide 2 → image 2.
3. DO NOT reuse images.
4. If slides > images, omit image_path.
5. image_path MUST be exactly one of the provided paths.

JSON SCHEMA:
[
  {
    "title": "Slide Title",
    "content": ["bullet 1", "bullet 2"],
    "image_path": "images/example.png"
  }
]
"""

CR_SYSTEM_PROMPT = """
You are a highly skilled code reviewer. Your role is to carefully analyze and evaluate code for correctness, clarity, 
and adherence to the given specifications. Your feedback should be precise, constructive, and well-structured.
"""

CR_USER_PROMPT = """
Here is the Slidev grammar specification: <SlidevGrammar> Additionally, here are two files: 
Text Summary: <TextSummary.md> 
Image Captions: <ImageCaption.md> 
Please review the code in <SlidevCode.md> to ensure:
1. It adheres to the Slidev grammar described in <Slidev Grammar>.
2. The content aligns with the information in <TextSummary.md> and <ImageCaption.md>.
3. **IMAGE PATH CHECK**: Verify that all image paths start with "images/" and do NOT use backslashes or absolute paths.
4. The code meets all content and layout requirements, including handling of images, text,and page structure as specified. 
Output your review as a Markdown file named <CodeReview.md>. Your review should clearly identify any errors or inconsistencies in the code, 
along with suggestions for improvement. If the code is correct, confirm that it meets all requirements.
"""

PR_SYSTEM_PROMPT = """
You are an expert visual page reviewer. Your role is to evaluate the layout and design of slides, 
ensuring they are visually appealing and properly aligned. Your feedback should be clear, actionable, 
and focused on improving the layout without altering the core content.
"""

PR_USER_PROMPT = """
Here are is a slide: <SlidevPages> 
Additionally, here are two supporting files: 
Image Information: <ImageCaption.md> 
Original Images: <Document Images> 
Please review each slide to check:
1. Whether any text or image exceeds the slide boundaries.
2. Whether the layout ensures a proper balance between text and images, avoiding over crowding or large empty spaces.
3. Whether the font sizes and styles are legible and consistent throughout the slide, ensuring readability without clashing with the visuals.
4. Whether the aspect ratios of images are preserved, and whether wide or tall images are placed appropriately without distorting the layout.
For each slide:
1. Indicate whether modifications are needed by answering with "yes" or "no".
2. If "yes", provide specific suggestions to adjust the positions of existing images. Do not add or remove any images. 
Output your review as a Markdown file named <PageReview.md>. Ensure your feedback is concise and easy to follow.
"""

# Also update the system prompt for the final agent
CGR_SYSTEM_PROMPT = "You are a specialized JSON data formatter. You never output Markdown or Slidev code. You only output structured JSON arrays."

# UPDATED: Enforce path rule in the correction phase too
# Force the final agent to stay in JSON mode
CGR_USER_PROMPT = """
Act as a JSON Architect. Convert the following text and image summaries into a valid JSON array for PowerPoint.

INPUTS:
Text Summary: <TextSummary.md> 
Image Captions: <ImageCaption.md> 
Review Feedback: <CodeReview.md> or <PageReview.md> 

CRITICAL RULES:
1. OUTPUT ONLY A VALID JSON ARRAY.
2. DO NOT include markdown backticks (```json).
3. DO NOT include headers like "# SlidevCode.md".
4. START with '[' and END with ']'.

SCHEMA:
[
  {
    "title": "Slide Title",
    "layout": "bullet_points",
    "content": ["point 1", "point 2"],
    "image_path": "images/filename.png"
  }
]
"""

# --- SLIDEV TEMPLATES ---
SLIDEV_GRAMMAR = """
# Slidev Markdown Grammar
---
# Frontmatter (first slide)
layout: cover
title: My Slide Title
---

# Standard Slide

# Slide Title

- Bullet point 1
- Bullet point 2

---
layout: two-cols

::left::
# Left Column
Content here

::right::
# Right Column
Content here

---
# Image Slide
![Alt Text](images/filename.png)
"""

SLIDEV_PAGES = """
Available Layouts:
1. 'cover': Use for the first slide. Contains title and subtitle.
2. 'default': Standard Title + Content.
3. 'two-cols': Split screen. Use ::left:: and ::right:: separators.
4. 'image-right': Content on left, image on right.
5. 'center': Centered text for quotes or impact statements.
"""