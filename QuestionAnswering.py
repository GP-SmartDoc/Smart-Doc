from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict, Annotated
import operator
import json
import re
from RAG import RAGEngine
from langchain_core.language_models import BaseChatModel

# ============================================================
# STATE DEFINITION
# ============================================================

class QuestionAnsweringGraphState(TypedDict):
    messages: Annotated[list, operator.add]
    llm_calls: Annotated[int, operator.add]

    user_question: str
    retrieved_text_chunks: list[str]
    retrieved_images: list[str]

    # ---- Hierarchical summaries ----
    text_chunk_summaries: list[str]          # A (Level 1)
    image_summaries: list[str]               # A (Level 1)

    text_summary: str                        # A (Level 2)
    image_summary: str                       # A (Level 2)

    cross_modal_analysis: dict               # D

    final_summary: dict                     # A (Level 3)


# ============================================================
# MODULE
# ============================================================

class QuestionAnsweringModule:

    def __init__(
        self,
        retriever: RAGEngine,
        model: BaseChatModel = ChatOllama(model="qwen3-vl:4b", temperature=0)
    ):
        self.retriever = retriever
        self.model = model
        self.workflow = StateGraph(QuestionAnsweringGraphState)

        # ---------------- Utility ----------------

        def clean_json(s: str):
            s = re.sub(r"```json|```", "", s)
            return s.strip()

        # ============================================================
        # A1) MICRO TEXT SUMMARIZATION (PER CHUNK)
        # ============================================================

        def text_micro_summarizer(state):
            summaries = []

            for chunk in state["retrieved_text_chunks"]:
                resp = self.model.invoke([
                    SystemMessage(
                        content="""
                        Summarize the following text chunk.
                        Extract only factual information relevant to the question.
                        Keep it concise.
                        """
                    ),
                    HumanMessage(
                        content=f"""
                        Question: {state["user_question"]}
                        Text Chunk: {chunk}
                        """
                    )
                ])
                summaries.append(resp.content)

            return {
                "text_chunk_summaries": summaries,
                "llm_calls": len(summaries)
            }

        # ============================================================
        # A1) MICRO IMAGE SUMMARIZATION (PER IMAGE)
        # ============================================================

        def image_micro_summarizer(state):
            summaries = []

            for img in state["retrieved_images"]:
                resp = self.model.invoke([
                    SystemMessage(
                        content="""
                        Analyze the image and extract factual information.
                        Focus on diagrams, labels, relationships, and visual structure.
                        """
                    ),
                    HumanMessage(
                        content=f"""
                        Question: {state["user_question"]}
                        Image (base64): {img}
                        """
                    )
                ])
                summaries.append(resp.content)

            return {
                "image_summaries": summaries,
                "llm_calls": len(summaries)
            }

        # ============================================================
        # A2 + B) MODALITY-LEVEL SUMMARIZATION (STRUCTURED)
        # ============================================================

        def modality_text_summary(state):
            resp = self.model.invoke([
                SystemMessage(
                    content="""
                    Merge the following text chunk summaries.
                    Remove redundancy.
                    Return a JSON with key factual points.
                    """
                ),
                HumanMessage(
                    content=json.dumps({
                        "question": state["user_question"],
                        "chunk_summaries": state["text_chunk_summaries"]
                    })
                )
            ])

            return {
                "text_summary": resp.content,
                "llm_calls": 1
            }

        def modality_image_summary(state):
            resp = self.model.invoke([
                SystemMessage(
                    content="""
                    Merge the following image summaries.
                    Focus on visual-only insights.
                    Return a JSON with key visual facts.
                    """
                ),
                HumanMessage(
                    content=json.dumps({
                        "question": state["user_question"],
                        "image_summaries": state["image_summaries"]
                    })
                )
            ])

            return {
                "image_summary": resp.content,
                "llm_calls": 1
            }

        # ============================================================
        # D) CROSS-MODAL ANALYSIS
        # ============================================================

        def cross_modal_reasoning(state):
            resp = self.model.invoke([
                SystemMessage(
                    content="""
                    Compare text and image summaries.
                    Identify:
                    - Overlapping facts
                    - Text-only facts
                    - Image-only facts
                    - Any contradictions
                    Return structured JSON.
                    """
                ),
                HumanMessage(
                    content=json.dumps({
                        "text_summary": state["text_summary"],
                        "image_summary": state["image_summary"]
                    })
                )
            ])

            return {
                "cross_modal_analysis": json.loads(clean_json(resp.content)),
                "llm_calls": 1
            }

        # ============================================================
        # A3 + B + C) FINAL MULTIMODAL SUMMARIZER
        # ============================================================

        def final_summarizer(state):
            resp = self.model.invoke([
                SystemMessage(
                    content="""
                    You are a multimodal summarization agent.

                    Step 1: Identify the most important facts.
                    Step 2: Resolve overlaps and contradictions.
                    Step 3: Produce a concise, accurate final summary.

                    Return JSON ONLY in this format:
                    {
                      "summary": "...",
                      "text_based_points": [...],
                      "image_based_points": [...],
                      "cross_modal_insights": [...]
                    }
                    """
                ),
                HumanMessage(
                    content=json.dumps({
                        "question": state["user_question"],
                        "text_summary": state["text_summary"],
                        "image_summary": state["image_summary"],
                        "cross_modal_analysis": state["cross_modal_analysis"]
                    })
                )
            ])

            return {
                "final_summary": json.loads(clean_json(resp.content)),
                "llm_calls": 1
            }

        # ============================================================
        # GRAPH WIRING
        # ============================================================

        self.workflow.add_node("text_micro", text_micro_summarizer)
        self.workflow.add_node("image_micro", image_micro_summarizer)
        self.workflow.add_node("text_summary", modality_text_summary)
        self.workflow.add_node("image_summary", modality_image_summary)
        self.workflow.add_node("cross_modal", cross_modal_reasoning)
        self.workflow.add_node("final", final_summarizer)

        self.workflow.add_edge(START, "text_micro")
        self.workflow.add_edge(START, "image_micro")

        self.workflow.add_edge("text_micro", "text_summary")
        self.workflow.add_edge("image_micro", "image_summary")

        self.workflow.add_edge("text_summary", "cross_modal")
        self.workflow.add_edge("image_summary", "cross_modal")

        self.workflow.add_edge("cross_modal", "final")
        self.workflow.add_edge("final", END)

        self.app = self.workflow.compile()

    # ============================================================
    # INVOKE
    # ============================================================

    def invoke(self, question: str):
        retrieved = self.retriever.query(question, k_text=6, k_image=4)

        state = {
            "messages": [],
            "llm_calls": 0,
            "user_question": question,
            "retrieved_text_chunks": retrieved["text"],
            "retrieved_images": retrieved["images"]
        }

        result = self.app.invoke(state)
        return result["final_summary"]

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
You are tasked with summarizing and evaluating the collective responses provided by multiple agents.

You have access to:
- Answers: the individual answers from all agents.

Your task consists of the following stages:

--- ANALYSIS STAGE ---
Analyze the provided answers with the following constraints:

(A) Redundancy & Semantic Clustering:
- Identify semantically similar or overlapping ideas across different agents.
- Merge repeated ideas into a single unified point.
- Avoid repeating the same concept using different wording.

(B) Structure-Aware Reasoning:
- Identify the logical role of each idea (e.g., definition, argument, evidence, limitation, conclusion).
- Preserve a coherent structure when forming the final reasoning.

(C) Consistency & Quality Evaluation:
- Evaluate each answer for correctness, relevance, and internal consistency.
- Identify contradictions, gaps, or weak reasoning among agents.
- Prefer ideas supported by multiple agents or stronger reasoning.

(D) Faithfulness Constraint:
- Use ONLY information explicitly stated in the agents’ answers.
- Do NOT introduce new facts, assumptions, or external knowledge.
- If important information is missing or agents disagree, explicitly acknowledge uncertainty.

--- SYNTHESIS STAGE ---
Synthesize the most accurate and reliable information by:
- Selecting the strongest merged ideas after clustering.
- Resolving conflicts by favoring better-supported or clearer reasoning.
- Discarding redundant, weak, or unsupported claims.

--- CONCLUSION STAGE ---
Produce a final answer that:
- Reflects agent consensus when it exists.
- Otherwise, presents the most credible and well-supported conclusion.
- Is concise, non-redundant, and clearly reasoned.

--- OUTPUT FORMAT ---
Return ONLY the final result in the following JSON format:
{"Answer": "<final synthesized answer>"}

Do not include explanations, analysis steps, or any additional text.
"""
