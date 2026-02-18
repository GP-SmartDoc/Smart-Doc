from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage

import config.slide_generation_prompts as prompts
from config.model import model
import os

from utils.image import encode_image_from_path

def Image_Captioner(state: dict):
    # 1. Get Image Filenames to show the LLM
    retrieved_imgs = state.get("retrieved_images", [])
    
    # Create a string list of filenames: "Image 1: chart.png", etc.
    filename_list = []
    for i, path in enumerate(retrieved_imgs):
        filename = os.path.basename(path)
        filename_list.append(f"Image {i+1} Filename: {filename}")
    
    filenames_str = "\n".join(filename_list)

    # 2. Update the prompt to include these filenames
    text_prompt = prompts.IC_USER_PROMPT.replace("<Document Text>", state["retrieved_text"])
    text_prompt += f"\n\nREFERENCED IMAGE FILENAMES:\n{filenames_str}"
    
    # 3. Build the message (using the fix from before)
    message_content = [{"type": "text", "text": text_prompt}]
    
    for img_path in retrieved_imgs:
        # Use the helper to get base64 (from previous fix)
        base64_image = encode_image_from_path(img_path)
        if base64_image:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

    # 4. Invoke LLM
    agent_answer: AIMessage = model.invoke([
        SystemMessage(content=prompts.IC_SYSTEM_PROMPT),
        HumanMessage(content=message_content)
    ])
    
    return {
        "messages": [agent_answer],
        "llm_calls": 1,
        "Image_Captioner_output": agent_answer.content
    }