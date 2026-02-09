from langchain.tools import tool
#from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import AnyMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict, Annotated
import operator
from langchain.messages import SystemMessage
from langchain.messages import ToolMessage
from typing import Literal
from langchain_core.language_models import BaseChatModel
import json
import re
import base64
import os
import shutil
from RAG import RAGEngine

class SlideGenerationGraphState(TypedDict):
    messages:Annotated[list[str], operator.add]
    llm_calls:int

    #user_question:str
    retrieved_text:str
    retrieved_images:list[str] # <-- ?????

    Text_Summarizer_output:str
    Image_Captioner_output:str
    Code_Generator_output:str # without review
    Code_Reviewer_output:str
    Page_Reviewer_output:str
    Code_Generator_output_Reviewed:str
    
class SlideGenerationModule():
    def __init__(self, retriever:RAGEngine, model:BaseChatModel=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")):
        self.__retriever = retriever
        self.__model = model
        workflow = StateGraph(SlideGenerationGraphState)

        workflow.add_node("Text_Summarizer", self.Text_Summarizer)
        workflow.add_node("Image_Captioner", self.Image_Captioner)
        workflow.add_node("Code_Generator", self.Code_Generator)
        workflow.add_node("Code_Reviewer", self.Code_Reviewer)
        workflow.add_node("Page_Reviewer", self.Page_Reviewer)
        workflow.add_node("Code_Generator_Reviewed", self.Code_Generator_Reviewed)
        
        workflow.add_edge(START, "Text_Summarizer")
        workflow.add_edge("Text_Summarizer", "Image_Captioner")
        workflow.add_edge("Image_Captioner", "Code_Generator")
        workflow.add_edge("Code_Generator", "Code_Reviewer")
        #workflow.add_edge("Code_Reviewer", "Page_Reviewer")
        #workflow.add_edge("Page_Reviewer", "Code_Generator_Reviewed")
        workflow.add_edge("Code_Reviewer", "Code_Generator_Reviewed")
        workflow.add_edge("Code_Generator_Reviewed", END)
        
        self.__workflow = workflow.compile()
        
        
    def Text_Summarizer(self, state:dict)   :
        agent_answer:AIMessage = self.__model.invoke(
            [
                SystemMessage(
                        content=TS_SYSTEM_PROMPT
                    ),
                HumanMessage(
                        content=TS_USER_PROMPT.replace(
                            "<Document text>",
                            state["retrieved_text"]
                        )
                    )
            ]
        )

        self._log_agent_output("Text_Summarizer", agent_answer.content)

        return {
            "messages": [agent_answer.content],
            "llm_calls": state.get('llm_calls', 0) + 1,
            "Text_Summarizer_output": agent_answer.content
        }
        
    def Image_Captioner(self, state: dict):
        # 1. Get Image Filenames to show the LLM
        retrieved_imgs = state.get("retrieved_images", [])
        
        # Create a string list of filenames: "Image 1: chart.png", etc.
        filename_list = []
        for i, path in enumerate(retrieved_imgs):
            filename = os.path.basename(path)
            filename_list.append(f"Image {i+1} Filename: {filename}")
        
        filenames_str = "\n".join(filename_list)

        # 2. Update the prompt to include these filenames
        text_prompt = IC_USER_PROMPT.replace("<Document Text>", state["retrieved_text"])
        text_prompt += f"\n\nREFERENCED IMAGE FILENAMES:\n{filenames_str}"
        
        # 3. Build the message (using the fix from before)
        message_content = [{"type": "text", "text": text_prompt}]
        
        for img_path in retrieved_imgs:
            # Use the helper to get base64 (from previous fix)
            base64_image = self._encode_image(img_path)
            if base64_image:
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

        # 4. Invoke LLM
        agent_answer: AIMessage = self.__model.invoke([
            SystemMessage(content=IC_SYSTEM_PROMPT),
            HumanMessage(content=message_content)
        ])

        self._log_agent_output("Image_Captioner", agent_answer.content)
        
        return {
            "messages": [agent_answer.content],
            "llm_calls": state.get('llm_calls', 0) + 1,
            "Image_Captioner_output": agent_answer.content
        }
        
    def Code_Generator(self, state:dict)   :
        agent_answer:AIMessage = self.__model.invoke(
            [
                SystemMessage(
                        content=CG_SYSTEM_PROMPT
                    ),
                HumanMessage(
                        content=CG_USER_PROMPT.replace(
                            "<SlidevGrammar>",
                            SLIDEV_GRAMMAR
                        ).replace(
                            "<TextSummary.md>",
                            state["Text_Summarizer_output"]
                        ).replace(
                            "<ImageCaption.md>",
                            state["Image_Captioner_output"]
                        )
                )
            ]
        )

        self._log_agent_output("Code_Generator", agent_answer.content)

        return {
            "messages": [agent_answer.content],
            "llm_calls": state.get('llm_calls', 0) + 1,
            "Code_Generator_output": agent_answer.content
        }
        
    def Code_Reviewer(self, state:dict)   :
        agent_answer:AIMessage = self.__model.invoke(
            [
                SystemMessage(
                        content=CR_SYSTEM_PROMPT
                    ),
                HumanMessage(
                        content=CR_USER_PROMPT.replace(
                            "<SlidevGrammar>",
                            SLIDEV_GRAMMAR
                        ).replace(
                            "<TextSummary.md>",
                            state["Text_Summarizer_output"]
                        ).replace(
                            "<ImageCaption.md>",
                            state["Image_Captioner_output"]
                        ).replace(
                            "<SlidevCode.md>",
                            state["Code_Generator_output"]
                        )
                )
            ]
        )

        self._log_agent_output("Code_Reviewer", agent_answer.content)

        return {
            "messages": [agent_answer.content],
            "llm_calls": state.get('llm_calls', 0) + 1,
            "Code_Reviewer_output": agent_answer.content
        }
        
    def Page_Reviewer(self, state:dict)   :
        agent_answer:AIMessage = self.__model.invoke(
            [
                SystemMessage(
                        content=PR_SYSTEM_PROMPT
                    ),
                HumanMessage(
                        content=PR_USER_PROMPT.replace(
                            "<SlidevPages>",
                            SLIDEV_PAGES
                        ).replace(
                            "<ImageCaption.md>",
                            state["Image_Captioner_output"]
                        ).replace(
                            "<Document Images>",
                            "\n".join(state["retrieved_images"])
                        )
                )
            ]
        )

        self._log_agent_output("Page_Reviewer", agent_answer.content)

        return {
            "messages": [agent_answer.content],
            "llm_calls": state.get('llm_calls', 0) + 1,
            "Page_Reviewer_output": agent_answer.content
        }
        
    def Code_Generator_Reviewed(self, state:dict)   :
        agent_answer:AIMessage = self.__model.invoke(
            [
                SystemMessage(
                        content=CGR_SYSTEM_PROMPT
                    ),
                HumanMessage(
                        content=CGR_USER_PROMPT.replace(
                            "<SlidevGrammar>",
                            SLIDEV_GRAMMAR
                        ).replace(
                            "<TextSummary.md>",
                            state["Text_Summarizer_output"]
                        ).replace(
                            "<ImageCaption.md>",
                            state["Image_Captioner_output"]
                        ).replace(
                            "<CodeReview.md>",
                            state["Code_Reviewer_output"]
                        )
                )
            ]
        )

        self._log_agent_output("Code_Generator_Reviewed", agent_answer.content)

        return {
            "messages": [agent_answer.content],
            "llm_calls": state.get('llm_calls', 0) + 1,
            "Code_Generator_output_Reviewed": agent_answer.content
        }
        
    def _log_agent_output(self, agent_name, content):
        """Appends agent output to a debug log file."""
        with open("workflow_debug_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*20} AGENT: {agent_name} {'='*20}\n")
            f.write(content)
            f.write(f"\n\n")
            
        
    # In SlideGeneration.py inside SlideGenerationModule class

    def generate_slides(self, topic: str):
        """
        Main entry point: Queries RAG, copies images to local folder, then runs generation.
        """
        print(f"Retrieving content for: {topic}...")
        rag_result = self.__retriever.query(topic, k_text=5, k_image=5)

        with open("workflow_debug_log.txt", "w", encoding="utf-8") as f:
            f.write(f"--- Workflow Log for Topic: {topic} ---\n")
        
        text_content = "\n\n".join(rag_result.get("text", []))
        original_images = rag_result.get("images", [])
        
        # --- CRITICAL FIX: Copy images to local 'images/' folder ---
        # This fixes the absolute path/backslash error in Slidev
        local_image_paths = []
        
        if original_images:
            os.makedirs("images", exist_ok=True) # Create folder
            print(f"Copying {len(original_images)} images to local 'images/' folder...")
            
            for src_path in original_images:
                if os.path.exists(src_path):
                    filename = os.path.basename(src_path)
                    # Sanitize filename (remove spaces, etc. if needed)
                    filename = filename.replace(" ", "_")
                    
                    dst_path = os.path.join("images", filename)
                    shutil.copy(src_path, dst_path)
                    
                    # Store as FORWARD SLASH path for Markdown
                    # e.g., "images/my_diagram.png"
                    local_image_paths.append(f"images/{filename}")
                else:
                    print(f"Warning: Source image not found: {src_path}")
        
        if not text_content:
            text_content = "No specific text found in documents."
            print("Warning: No text retrieved from RAG.")

        initial_state = {
            "messages": [HumanMessage(content=topic)],
            "llm_calls": 0,
            "retrieved_text": text_content,
            "retrieved_images": local_image_paths, # Pass the CLEAN local paths
            "Text_Summarizer_output": "",
            "Image_Captioner_output": "",
            "Code_Generator_output": "",
            "Code_Reviewer_output": "",
            "Page_Reviewer_output": "",
            "Code_Generator_output_Reviewed": ""
        }
        
        print("Starting Multi-Agent Workflow...")
        final_state = self.__workflow.invoke(initial_state)
        return final_state["Code_Generator_output_Reviewed"]
    
    def _encode_image(self, image_path):
        """Helper to check if valid file and convert to base64"""
        if not image_path or not isinstance(image_path, str):
            print(f"Warning: Invalid image path: {image_path}")
            return None
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            return None
            
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
        
    def visualize_workflow(self):
        from IPython.display import Image, display
        display(Image(self.__workflow.get_graph().draw_mermaid_png()))
        
        
        
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
CG_USER_PROMPT = """
Here is a Slidev grammar example file: <SlidevGrammar> Additionally, here are two files:
Text Summary: <TextSummary.md> 
Image Captions: <ImageCaption.md>
Please merge the content of <TextSummary.md> and <ImageCaption.md> into a single file formatted using the Slidev grammar provided.

The merged file should meet the following requirements:
1. **CRITICAL IMAGE PATH RULE**: 
   - If using `layout: image-right`, `layout: image-left`, or `layout: full`, you must define the image in the slide frontmatter like this: `image: images/filename.png`.
   - If using a standard layout, use standard markdown: `![Title](images/filename.png)`.
   - DO NOT use absolute paths. Use forward slashes (/) only.
2. Design each page so that elements in the same column are not overcrowded. Split content into multiple columns if necessary to prevent overflow.
3. Avoid pages with too few elements. Expand content where needed to ensure an appropriate balance.
4. If certain columns contain only images without text, center the images on the page.
5. Consider the aspect ratio of images:
6. If the aspect ratio exceeds 2:1, the image should span multiple columns in multi-column layouts rather than appearing in a single column.
7. The first page should include the title and author information.
8. The last page should serve as a summary page. 
Output the final code in Markdown format as <SlidevCode.md>. Ensure the code is clean, adheres to the Slidev grammar, and satisfies all specified layout requirements.
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

CGR_SYSTEM_PROMPT = CG_SYSTEM_PROMPT

# UPDATED: Enforce path rule in the correction phase too
CGR_USER_PROMPT = """
Here is the Slidev grammar specification: <SlidevGrammar>
Additionally, here are three files: 
Text Summary: <TextSummary.md> 
Image Captions: <ImageCaption.md> 
Review Feedback: <CodeReview.md>

Please merge the content of <TextSummary.md> and <ImageCaption.md> into a single file formatted using the Slidev grammar provided. 
While doing so, take into account the feedback provided in <CodeReview.md> and address any relevant issues.

The merged file should meet the following requirements:
1. **CRITICAL**: Ensure all image paths are relative and use forward slashes (e.g., `images/filename.png`). Correct any paths that use backslashes or absolute paths.
2. Use the image descriptions from <ImageCaption.md> as the definitive source for image content.
3. Design each page so that elements in the same column are not overcrowded. Split content into multiple columns if necessary to prevent overflow.
4. Avoid pages with too few elements. Expand content where needed to ensure an appropriate balance.
5. If certain columns contain only images without text, center the images on the page.
6. Consider the aspect ratio of images:
7. If the aspect ratio exceeds 2:1, the image should span multiple columns in multi-column layouts rather than appearing in a single column.
8. The first page should include the title and author information.
9. The last page should serve as a summary page. 
Output the final code in Markdown format as <SlidevCode.md>. Ensure the code is clean, adheres to the Slidev grammar, and satisfies all specified layout requirements.
""" 

# --- SLIDEV TEMPLATES ---
SLIDEV_GRAMMAR = """
# Slidev Markdown Grammar Reference

## 1. Frontmatter (Global Configuration)
Always start the file with this block.
---
theme: seriph
background: background\simple-blue-gradient-background-vector-business_53876-169287.jpg
class: text-center
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
title: [Insert Presentation Title]
---

## 2. Standard Slide Syntax
---
# [Slide Title]

- [Bullet Point 1]
- [Bullet Point 2]
- **Bold Text** for emphasis

---

## 3. Two-Column Layout (Text + Text)
Use this for comparing two concepts.
---
layout: two-cols

::left::
# Left Side Title
- Point A
- Point B

::right::
# Right Side Title
- Point C
- Point D
---

## 4. Image + Text Layouts (Preferred over manual images)
Use `image-right` or `image-left` layouts. Define the image path in the frontmatter of the slide.

---
layout: image-right
image: images/filename.png
backgroundSize: contain
---
# Slide Title

- This text appears on the left.
- The image appears on the right automatically.
---

## 5. Full Screen Image
---
layout: full
image: images/filename.png
---
---

## 6. Quote / Statement Slide
Use for impact statements or summaries.
---
layout: center
class: text-center
---
# "Key takeaway or quote goes here"

- Author Name
---

## 7. Section Break / Intro
---
layout: intro
class: text-center
---
# [Section Title]
[Subtitle or short description]
---

## 8. Styling Tips (Tailwind CSS)
You can use HTML tags with Tailwind classes for coloring or sizing.
- <span class="text-red-500">Red Text</span>
- <span class="text-xl font-bold">Big Bold Text</span>
- <div class="grid grid-cols-3 gap-4"> (Advanced Grid) </div>
"""

SLIDEV_PAGES = """
Layout Usage Guide:

1. **layout: cover** - USE ONLY ONCE: For the very first slide. 
   - Must include title and author.

2. **layout: intro**
   - Use to introduce a major new section or topic change.
   - Centers the text for dramatic effect.

3. **layout: default**
   - Use for standard bullet points.
   - Best when there are no images, or small inline images only.

4. **layout: image-right** (Highly Recommended)
   - Use when you have a specific image to discuss.
   - Puts text on the left, image on the right.
   - Looks more professional than manually inserting `![img]()`.

5. **layout: image-left**
   - Same as above, but image on the left.
   - Use to vary the visual rhythm.

6. **layout: two-cols**
   - Use when comparing two distinct lists or concepts.
   - Do NOT use this just to place an image (use image-right instead).

7. **layout: center**
   - Use for the "Summary" slide or a powerful Quote.
   - Centers content horizontally and vertically.

8. **layout: full**
   - Use for high-resolution diagrams that need maximum space.
   - No text is preferable, or very minimal overlay text.
"""