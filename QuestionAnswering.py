from langchain.tools import tool
from langchain_ollama import ChatOllama
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
from RAG import RAGEngine

class QuestionAnsweringGraphState(TypedDict):
    messages:Annotated[list[str], operator.add]
    llm_calls:int

    user_question:str
    retrieved_text:str
    retrieved_images:list[str] # <-- ?????

    ga_output:str
    ca_output:str
    ta_output:str
    ia_output:str
    sa_output:str

class QuestionAnsweringModule():
    def __init__(self, retriever:RAGEngine, model:BaseChatModel = ChatOllama(model="qwen3:8b",temperature=0)):
        self.__retriever = retriever
        self.__model = model
        self.__workflow_builder = StateGraph(QuestionAnsweringGraphState)

        def clean_json_string(s):
            s = re.sub(r'```json\s*', '', s)
            s = re.sub(r'```', '', s)
            return s.strip()
        
        def _general_agent(state:dict):
            agent_answer:AIMessage = self.__model.invoke(
                [
                    SystemMessage(
                        content=GA_SYSTEM_PROMPT
                    ),
                    HumanMessage(
                        content=f"""
                            Textual Content: {state.get("retrieved_text", "No text provided")}
                            Image Content: {state.get("retrieved_images", "No images provided")}
                            Question: {state.get("user_question", "")}
                        """
                    )
                ]
            )
                    
            return {
                "messages": [agent_answer],
                "llm_calls": state.get('llm_calls', 0) + 1,
                "ga_output": agent_answer.content
            }

        def _critical_agent(state:dict):
            agent_answer:AIMessage = self.__model.invoke(
                [
                    SystemMessage(
                        content=CA_SYSTEM_PROMPT
                    ),
                    HumanMessage(
                        content=f"""
                            Question: {state.get("user_question", "")}
                            Preliminary Answer: {state.get("ga_output"), ""}
                            Textual Content: {state.get("retrieved_text", "No text provided")}
                            Image Content: {state.get("retrieved_images", "No images provided")}
                        """
                    )
                ]
            )
                    
            return {
                "messages": [agent_answer],
                "llm_calls": state.get('llm_calls', 0) + 1,
                "ca_output": agent_answer.content
            }

        def _text_agent(state:dict):
            # The critical agents returns a json
            critical_agent_output = json.loads(clean_json_string(state.get("ca_output", "")))
            critical_text_info = critical_agent_output.get("text")
            agent_answer:AIMessage = self.__model.invoke(
                [
                    SystemMessage(
                        content=TA_SYSTEM_PROMPT
                    ),
                    HumanMessage(
                        content=f"""
                            Question: {state.get("user_question", "")}
                            Critical Text Information: {critical_text_info}
                            Textual Content: {state.get("retrieved_text", "No text provided")}
                        """
                    )
                ]
            )
                    
            return {
                "messages": [agent_answer],
                "llm_calls": state.get('llm_calls', 0) + 1,
                "ta_output": agent_answer.content
            }

        def _image_agent(state:dict):
            # The critical agents returns a json
            critical_agent_output = json.loads(clean_json_string(state.get("ca_output", "")))
            critical_image_info = critical_agent_output.get("image")
            
            msg_content = [
                {"type": "text", "text": f"Textual Content: {state.get('retrieved_text')}\nQuestion: {state.get('user_question')}"}
            ]
            if state.get("retrieved_images"):
                for img_b64 in state.get("retrieved_images"):
                    msg_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    })
                    
            agent_answer:AIMessage = self.__model.invoke(
                [
                    SystemMessage(
                        content=IA_SYSTEM_PROMPT
                    ),
                    HumanMessage(
                        content=f"""
                            Question: {state.get("user_question", "")}
                            Critical Image Information: {critical_image_info}
                            Image Content: {state.get("retrieved_images", "No images provided")}
                        """
                    )
                ]
            )

            return {
                "messages": [agent_answer],
                "llm_calls": state.get('llm_calls', 0) + 1,
                "ia_output": agent_answer.content
            }


        def _summarizing_agent(state:dict):

            agent_answer:AIMessage = self.__model.invoke(
                [
                    SystemMessage(
                        content=SA_SYSTEM_PROMPT
                    ),
                    HumanMessage(
                        content=f"""
                            Question: {state.get("user_question", "")}
                            Preliminary Answer: {state.get("ga_output", "")}
                            Text Agent Answer: {state.get("ta_output", "")}
                            Image Agent Answer: {state.get("ia_output", "")}
                        """
                    )
                ]
            )
                    
            return {
                "messages": [agent_answer],
                "llm_calls": state.get('llm_calls', 0) + 1,
                "sa_output": agent_answer.content
            }
        
        self.__workflow_builder.add_node("general_agent", _general_agent)
        self.__workflow_builder.add_node("critical_agent", _critical_agent)
        self.__workflow_builder.add_node("text_agent", _text_agent)
        self.__workflow_builder.add_node("image_agent", _image_agent)
        self.__workflow_builder.add_node("summarizing_agent", _summarizing_agent)

        self.__workflow_builder.add_edge(START, "general_agent")
        self.__workflow_builder.add_edge("general_agent", "critical_agent")
        self.__workflow_builder.add_edge("critical_agent", "text_agent")
        self.__workflow_builder.add_edge("critical_agent", "image_agent")
        self.__workflow_builder.add_edge("text_agent", "summarizing_agent")
        self.__workflow_builder.add_edge("image_agent", "summarizing_agent")
        self.__workflow_builder.add_edge("summarizing_agent", END)
        
        self.__workflow = self.__workflow_builder.compile()

    def invoke_workflow(self, prompt:str)->str:
        retrieved_data = self.__retriever.query(prompt)
        
        text_content = retrieved_data.get("text")
        if not text_content:
            text_content = "No relevant textual documents found."
            
        images_b64 = []
        for path in retrieved_data["images"]: 
            with open(path, "rb") as img_f: # check if file exists?
                b64 = base64.b64encode(img_f.read()).decode('utf-8')
                images_b64.append(b64)
                
        retrieved_text = "".join(text_content)
        retrieved_images = images_b64

        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "user_question": prompt,  
            "retrieved_text": retrieved_text, 
            "retrieved_images": retrieved_images,
            "llm_calls": 0
        }
        final_state = self.__workflow.invoke(initial_state)
        return final_state.get("sa_output", "")
    
    def visualize_workflow(self):
        from IPython.display import Image, display
        display(Image(self.__workflow.get_graph().draw_mermaid_png()))
        

GA_SYSTEM_PROMPT = """  
You are an advanced agent capable of analyzing both text and images. Your task is to use both the textual and visual information provided to answer the user’s question accurately.
Extract Text from Both Sources: If the image contains text, extract it using OCR, and consider both the text in the image and the provided textual content.
Analyze Visual and Textual Information: Combine details from both the image (e.g., objects, scenes, or patterns) and the text to build a comprehensive understanding of the content.
Provide a Combined Answer: Use the relevant details from both the image and the text to provide a clear, accurate, and context-aware response to the user's question.
When responding:
If both the image and text contain similar or overlapping information, cross-check and use both to ensure consistency.
If the image contains information not present in the text, include it in your response if it is relevant to the question.
If the text and image offer conflicting details, explain the discrepancies and clarify the most reliable source.
Since you have access to both text and image data, you can provide a more comprehensive answer than agents with single-source data.
"""

CA_SYSTEM_PROMPT = """
Provide a Python dictionary of 2 keypoints which you need for the question based on all given information. One is for text, the other is for image.
Respond exclusively in valid Dictionary of str format without any other text. For example, the format shold be: {"text": "keypoint for text", "image": "keypoint for image"}.
"""

TA_SYSTEM_PROMPT = """
You are a text analysis agent. Your job is to extract key information from the text and use it to answer the user’s question accurately. Here are the steps to follow:
Extract key details: Focus on the most important facts, data, or ideas related to the question.
Understand the context: Pay attention to the meaning and details.
Provide a clear answer: Use the extracted information to give a concise and relevant response to user's question.
Remeber you can only get the information from the text provided, so maybe other agents can help you with the image information.
"""

IA_SYSTEM_PROMPT = """
You are an advanced image processing agent specialized in analyzing and extracting information from images. The images may include document screenshots, illustrations, or photographs. Your primary tasks include:
Extracting textual information from images using Optical Character Recognition (OCR).
Analyzing visual content to identify relevant details (e.g., objects, patterns, scenes).
Combining textual and visual information to provide an accurate and context-aware answer to user's question.
Remeber you can only get the information from the images provided, so maybe other agents can help you with the text information.
"""

SA_SYSTEM_PROMPT = """
You are tasked with summarizing and evaluating the collective responses provided by multiple agents. You have access to the following information:
Answers: The individual answers from all agents.
Using this information, perform the following tasks:
Analyze: Evaluate the quality, consistency, and relevance of each answer. Identify commonalities, discrepancies, or gaps in reasoning.
Synthesize: Summarize the most accurate and reliable information based on the evidence provided by the agents and their discussions.
Conclude: Provide a final, well-reasoned answer to the question or task. Your conclusion should reflect the consensus (if one exists) or the most credible and well-supported answer.
Based on the provided answers from all agents, summarize the final decision clearly. You should only return the final answer in this dictionary format: {"Answer": <Your final answer here>}. Don't give other information.
"""


