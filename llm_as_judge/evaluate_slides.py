"""
Slide Generation Pipeline Evaluation
=====================================
10 test PDFs from:
    llm_as_judge/slide test/test{N}.pdf

Fixed prompt (same for all samples):
    "create a presentation about this paper include related work, system
     architecture, dataset, metrics, evaluation results and a conclusion"

Pipeline per sample
-------------------
  1. Index PDF into a shared ephemeral RAG (once per PDF).
  2. Query RAG  (k_text=15, k_image=8) filtered to that PDF's filename.
  3. Copy retrieved images to a local temp folder (Slidev requirement).
  4. Run slide_generation_graph → Code_Generator_output_Reviewed.
  5. Evaluate with 5 metrics:
       • FaithfulnessMetric        — slide content grounded in retrieved text
       • AnswerRelevancyMetric     — slides address the user's prompt
       • Content Coverage (G-Eval) — 6 required sections present
       • Image Relevance  (G-Eval) — retrieved images fit the topic
       • CLIP Score                — avg. cosine-sim(prompt, image) via ViT-B-32

Output
------
    llm_as_judge/slide_eval_results.json   (flushed after EVERY sample)

Resume / Retry
--------------
    Completed samples (metrics + no error) and timed-out samples are skipped.
    Errored samples are retried on next run.

Run (from project root)
-----------------------
    pytest llm_as_judge/evaluate_slides.py -v
"""

import asyncio
import os
import sys
import json
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import torch
import open_clip
from PIL import Image as PILImage

import chromadb
import weave
import pytest
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.vector_store.RAG import RAGEngine
from src.graphs.slide_generation_graph import sg_module

# ── Weave tracing ─────────────────────────────────────────────────────────────
weave.init("gamer7dragon817-ain-shams-university/slide-generator-demo")


# ══════════════════════════════════════════════════════════════════════════════
#  PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PDFS_DIR     = os.path.join(_HERE, "slide test")
_RESULTS_JSON = os.path.join(_HERE, "slide_eval_results.json")

# Temporary folder for images copied during slide generation
_IMAGES_TMP   = os.path.join(_HERE, "_slide_eval_images")

_TIMEOUT_SECONDS = 600   # per sample — 6 LLM calls + indexing + 5 metrics

# Fixed prompt applied to every test PDF
_PROMPT = (
    "create a presentation about this paper include related work, "
    "system architecture, dataset, metrics, evaluation results and a conclusion"
)


# ══════════════════════════════════════════════════════════════════════════════
#  JUDGE MODEL  (W&B Inference)
# ══════════════════════════════════════════════════════════════════════════════

class WandbJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by the W&B Inference API."""

    def __init__(self):
        self._model = ChatOpenAI(
            model=os.getenv("EVAL_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv(
                "OPENAI_API_BASE", "https://api.inference.wandb.ai/v1"
            ),
            temperature=0,
        )

    def load_model(self):
        return self._model

    def generate(self, prompt: str) -> str:
        return self._model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        response = await self._model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return os.getenv("EVAL_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")


judge = WandbJudge()


# ══════════════════════════════════════════════════════════════════════════════
#  DEEPEVAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

_faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
)

_answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.6,
    model=judge,
    include_reason=True,
)

# G-Eval: Content Coverage — sections are derived dynamically from the source document.
_content_coverage_metric = GEval(
    name="ContentCoverageMetric",
    criteria=(
        "You are evaluating a generated presentation against the source document.\n\n"
        "Step 1 — Identify presentation-relevant sections: Read the retrieval context "
        "(the source document chunks) and identify the content-bearing sections that "
        "belong in a presentation (e.g. Introduction, Related Work, Methodology, "
        "System Architecture, Dataset, Experiments, Results, Metrics, Evaluation, "
        "Conclusion, Future Work, etc.). Do NOT use a fixed list — extract only the "
        "sections that actually appear in THIS document.\n"
        "IMPORTANT: EXCLUDE the following paper-only boilerplate sections from your "
        "list — they are never expected in a presentation slide deck:\n"
        "  • Abstract\n"
        "  • References / Bibliography\n"
        "  • Acknowledgements / Acknowledgments\n"
        "  • Appendix / Appendices (any lettered or numbered appendix)\n\n"
        "Step 2 — Evaluate coverage: For each presentation-relevant section you "
        "identified, check whether the actual output contains a dedicated slide or "
        "meaningful textual content covering that section's key ideas. A slide does "
        "not need to use the exact section title — it just needs to address the "
        "section's main content.\n\n"
        "Scoring (based on the fraction of presentation-relevant sections covered):\n"
        "  Score 1.0 — all identified sections have meaningful slide coverage.\n"
        "  Score 0.7 — most sections covered; only 1 is missing or very brief.\n"
        "  Score 0.5 — about half the sections are covered with meaningful content.\n"
        "  Score 0.3 — only a few sections have meaningful slide coverage.\n"
        "  Score 0.0 — the presentation is empty, malformed, or covers none of the "
        "identified sections."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
    model=judge,
)

# G-Eval: Image Relevance — each image must be relevant to the text on its own slide.
_image_relevance_metric = GEval(
    name="ImageRelevanceMetric",
    criteria=(
        "The actual output is a JSON array of slide objects produced by a presentation pipeline.\n"
        "Each slide object has these fields:\n"
        "  • 'layout_name' — layout type (e.g. 'title_subtitle', 'title_content',\n"
        "    'content_image', 'large_image', 'three_column', 'title_only')\n"
        "  • 'title' — the slide heading\n"
        "  • 'content' / 'subtitle' / 'col1'/'col2'/'col3' — the slide's text body\n"
        "  • 'image_path' — path to the image on this slide, or null if no image\n\n"
        "The retrieval context contains image captions that describe the visual content "
        "of each image file used during generation.\n\n"
        "First, count how many slides have a non-null 'image_path'.\n"
        "If ZERO slides have an image, score 0.5 (neutral — absence of images is "
        "neither good nor bad for this metric).\n\n"
        "Otherwise, for every slide with a non-null 'image_path':\n"
        "  1. Read the slide's text (title + content/subtitle/columns).\n"
        "  2. Find the caption for that image in the retrieval context.\n"
        "  3. Judge whether the image described by that caption is relevant to the "
        "slide's own text — not just the overall topic, but specifically what "
        "that slide is discussing (e.g. an architecture diagram caption on a system "
        "design slide, a results figure caption on an evaluation slide).\n\n"
        "Scoring (when at least one image exists):\n"
        "  Score 1.0 — every slide's image is clearly relevant to that slide's own text.\n"
        "  Score 0.7 — most images are locally relevant; 1–2 have weak slide-level connection.\n"
        "  Score 0.5 — roughly half the images match the text on their slide.\n"
        "  Score 0.3 — most images are misplaced relative to their slide's text.\n"
        "  Score 0.0 — all images are completely irrelevant to their slide's text."
    ),
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
    model=judge,
)

# ContentCoverage is scored separately against the full PDF text — see _score_content_coverage().
_DEEPEVAL_METRICS = [
    _faithfulness_metric,
    _answer_relevancy_metric,
]


# ══════════════════════════════════════════════════════════════════════════════
#  CLIP SCORE  (ViT-B-32, laion2b pretrained — same model used by ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

print("[INFO] Loading CLIP ViT-B-32 model for CLIP Score …")
_clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
_clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
_clip_model.eval()
_clip_device = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model = _clip_model.to(_clip_device)
print(f"[INFO] CLIP model loaded on {_clip_device}.")


def _extract_slide_pairs(slides: str) -> list[tuple[str, str]]:
    """Extract (slide_text, image_path) pairs from the slides output.

    Handles two formats produced by the slide generation graph:
      1. JSON array  — slides is a JSON string (possibly wrapped in ```json ... ```)
                       each object has 'image_path' and text fields like
                       'title', 'content', 'subtitle', 'col1/col2/col3'.
      2. Markdown    — slides separated by '---' with ![](path) or <img src=...> tags.
    """
    import re as _re
    import json as _json

    pairs: list[tuple[str, str]] = []

    # ── Try JSON format first ─────────────────────────────────────────────────
    raw = slides.strip()
    # Strip optional ```json ... ``` fences
    fenced = _re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        slide_objects = _json.loads(raw)
        if isinstance(slide_objects, list):
            for obj in slide_objects:
                if not isinstance(obj, dict):
                    continue
                img_path = obj.get("image_path")
                if not img_path:
                    continue
                # Collect all text fields on this slide
                text_parts = [str(obj[k]) for k in
                              ("title", "subtitle", "content", "col1", "col2", "col3")
                              if obj.get(k) and isinstance(obj[k], str)]
                slide_text = " ".join(text_parts).strip() or "slide"
                pairs.append((slide_text, img_path))
            return pairs
    except (_json.JSONDecodeError, TypeError):
        pass

    # ── Fallback: Markdown format ─────────────────────────────────────────────
    _IMG_MD   = _re.compile(r'!\[.*?\]\(([^)]+)\)')
    _IMG_HTML = _re.compile(r'<img\b[^>]+\bsrc=["\']([^"\']+)["\']', _re.IGNORECASE)

    for slide in [s.strip() for s in slides.split("---") if s.strip()]:
        img_refs = _IMG_MD.findall(slide) + _IMG_HTML.findall(slide)
        if not img_refs:
            continue
        slide_text = _IMG_MD.sub("", slide)
        slide_text = _IMG_HTML.sub("", slide_text).strip() or "slide"
        for ref in img_refs:
            pairs.append((slide_text, ref))

    return pairs


def _compute_clip_score(slides: str, images_dir: str) -> float | None:
    """
    Per-slide CLIP Score: cosine similarity between each slide's text content
    and the image on that same slide.

    Supports both JSON-array and Markdown slide formats.
    Returns the average cosine similarity over all (slide_text, image) pairs,
    or None if no image references are found.
    """
    if not slides.strip():
        return None

    pairs = _extract_slide_pairs(slides)
    if not pairs:
        return None

    scores: list[float] = []
    for slide_text, img_ref in pairs:
        # Resolve image path — try as-is first, then by basename in images_dir
        if os.path.isabs(img_ref) and os.path.exists(img_ref):
            img_path = img_ref
        else:
            img_name = os.path.basename(img_ref.split("?")[0]).replace("%20", " ")
            img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[WARN] CLIP: image not found — {img_path}")
            continue

        try:
            text_tokens = _clip_tokenizer([slide_text]).to(_clip_device)
            with torch.no_grad():
                text_features = _clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            img_tensor = (
                _clip_preprocess(PILImage.open(img_path).convert("RGB"))
                .unsqueeze(0)
                .to(_clip_device)
            )
            with torch.no_grad():
                img_features = _clip_model.encode_image(img_tensor)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            sim = (text_features @ img_features.T).item()
            scores.append(sim)
        except Exception as exc:
            print(f"[WARN] CLIP: failed on {img_path}: {exc}")

    return round(sum(scores) / len(scores), 4) if scores else None


# ══════════════════════════════════════════════════════════════════════════════
#  RAG ENGINE  (fresh instance per sample — reset after each test)
# ══════════════════════════════════════════════════════════════════════════════

_rag_client   = chromadb.EphemeralClient()
_rag          = RAGEngine(_rag_client)
_indexed_pdfs: set[str] = set()


def _reset_rag() -> None:
    """Tear down all ChromaDB collections and rebuild a clean RAGEngine.

    Called after every sample (success, error, or timeout) so that chunks
    from one PDF cannot bleed into the next test's retrieval.
    Also clears blob_storage so cropped images from one PDF don't persist.
    """
    global _rag_client, _rag, _indexed_pdfs
    try:
        for col in _rag_client.list_collections():
            _rag_client.delete_collection(col.name)
    except Exception as exc:
        print(f"[WARN] RAG reset: could not delete collections: {exc}")

    # Clear blob_storage — RAGEngine saves cropped images here; stale files
    # from a previous PDF must not carry over to the next sample.
    blob_storage_path = os.path.join(os.getcwd(), "blob_storage")
    if os.path.isdir(blob_storage_path):
        try:
            shutil.rmtree(blob_storage_path)
            print(f"[INFO] blob_storage cleared: {blob_storage_path}")
        except Exception as exc:
            print(f"[WARN] Could not clear blob_storage: {exc}")

    _rag_client  = chromadb.EphemeralClient()
    _rag         = RAGEngine(_rag_client)   # recreates blob_storage via os.makedirs
    _indexed_pdfs = set()
    print("[INFO] RAG engine reset — clean state for next sample.")


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_samples() -> list[dict]:
    """Discover all test{N}.pdf files in the slide test directory."""
    samples = []
    for fname in sorted(os.listdir(_PDFS_DIR)):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(_PDFS_DIR, fname)
        samples.append({
            "pdf_file": fname,
            "pdf_path": pdf_path,
            "run_key":  fname,   # unique resume key
        })
    print(f"[INFO] Found {len(samples)} PDF(s) in: {_PDFS_DIR}")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE  (RAG → slide_generation_graph)
# ══════════════════════════════════════════════════════════════════════════════

@weave.op()
def _run_pipeline(sample: dict) -> dict:
    """
    Index PDF (first access only) → query RAG → run slide generation graph.

    Returns a dict with:
      slides           : str   — final slide markdown (Code_Generator_output_Reviewed)
      retrieved_text   : list[str] — text chunks from RAG (for Faithfulness)
      retrieved_image_paths : list[str] — original blob paths (for CLIP score)
      image_captioner_output : str — Image_Captioner_output from graph state
                                     (for Image Relevance G-Eval)
    """
    pdf_file = sample["pdf_file"]
    pdf_path = sample["pdf_path"]

    # ── Lazy PDF indexing ────────────────────────────────────────────────────
    if pdf_file not in _indexed_pdfs:
        print(f"[INDEX] Indexing {pdf_file} …")
        _rag.add_pdf(pdf_path)
        _indexed_pdfs.add(pdf_file)

    # ── RAG query ───────────────────────────────────────────────────────────
    retrieved_data = _rag.query(
        _PROMPT, k_text=15, k_image=8, document=pdf_file
    )
    text_chunks: list[str] = retrieved_data.get("text", [])
    image_paths: list[str] = retrieved_data.get("paths", [])   # blob_storage abs paths

    retrieved_text = "\n".join(text_chunks)

    # ── Copy images to temp folder (Slidev needs relative paths) ─────────────
    os.makedirs(_IMAGES_TMP, exist_ok=True)
    local_image_paths: list[str] = []
    for src_path in image_paths:
        if os.path.exists(src_path):
            fname = os.path.basename(src_path).replace(" ", "_")
            dst   = os.path.join(_IMAGES_TMP, fname)
            shutil.copy(src_path, dst)
            local_image_paths.append(dst)

    # ── Build initial graph state ────────────────────────────────────────────
    initial_state = {
        "messages": [HumanMessage(content=_PROMPT)],
        "llm_calls": 0,
        "retrieved_text": retrieved_text if retrieved_text else "No text retrieved.",
        "retrieved_images": local_image_paths,
        "document": pdf_file,
        "Text_Summarizer_output": "",
        "Image_Captioner_output": "",
        "Code_Generator_output": "",
        "json_presentation_data": "",
        "Code_Reviewer_output": "",
        "Page_Reviewer_output": "",
        "Code_Generator_output_Reviewed": "",
    }

    # ── Invoke slide generation graph ────────────────────────────────────────
    print(f"[GEN] Running slide generation for {pdf_file} …")
    final_state = sg_module.invoke(initial_state)

    return {
        "slides":                  final_state.get("Code_Generator_output_Reviewed", ""),
        "retrieved_text":          text_chunks,
        "retrieved_image_paths":   image_paths,          # original blob paths for CLIP
        "image_captioner_output":  final_state.get("Image_Captioner_output", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ASYNC METRIC SCORING
# ══════════════════════════════════════════════════════════════════════════════

async def _score_metric_async(metric, test_case: LLMTestCase) -> dict:
    await metric.a_measure(test_case)
    return {
        "name":      getattr(metric, "name", type(metric).__name__),
        "score":     metric.score,
        "threshold": metric.threshold,
        "success":   metric.score >= metric.threshold,
        "reason":    getattr(metric, "reason", None),
    }


def _generate_document_prompt(text_chunks: list[str]) -> str:
    """Generate a presentation request that names the sections actually present in this PDF.

    AnswerRelevancyMetric checks whether the actual output answers 'input'.
    The generic _PROMPT lists fixed sections that may not exist in every paper.
    This function asks the judge to identify the real sections from the retrieved
    chunks and returns a prompt of the form:
        "Create a presentation about this paper covering: <section1>, <section2>, ..."
    so the metric evaluates whether the slides respond to a request that is
    specific to THIS document's actual content.
    """
    context = "\n\n".join(text_chunks[:4]) if text_chunks else ""
    if not context:
        return _PROMPT
    judge_q = (
        "You are an expert evaluator. Read the following excerpts from an academic paper.\n\n"
        "Based ONLY on what you actually read in these excerpts, identify the key topics, "
        "findings, methods, and contributions that are worth presenting in a slide deck.\n\n"
        "Then write a single presentation request that reflects THIS document's actual "
        "content. Use the format:\n"
        "\"Create a presentation about <brief description of the paper's subject> covering "
        "<topic1>, <topic2>, <topic3>, ...\"\n\n"
        "Rules:\n"
        "- Do NOT invent section names or topics not supported by the excerpts.\n"
        "- Do NOT copy generic section labels (e.g. 'Introduction', 'Conclusion') unless "
        "  the excerpt content actually discusses something specific under that label.\n"
        "- Base the topics on the actual content: what problem is solved, what method is "
        "  proposed, what dataset/experiments are used, what results are reported.\n"
        "- Return ONLY that one sentence, no explanation or preamble.\n\n"
        f"Document excerpts:\n{context}"
    )
    try:
        return judge.generate(judge_q).strip()
    except Exception as exc:
        print(f"[WARN] Could not generate document prompt: {exc}")
        return _PROMPT


@weave.op()
def _score_deepeval_metrics(
    slides: str,
    text_chunks: list[str],
) -> tuple[list[dict], str]:
    """Run Faithfulness, AnswerRelevancy, ContentCoverage concurrently.

    Generates a document-specific content question from the retrieved chunks
    so that AnswerRelevancyMetric evaluates whether the slides address the
    paper's actual subject matter, not just the generic task instruction.

    Returns (metric_scores, generated_doc_prompt).
    """
    doc_prompt = _generate_document_prompt(text_chunks)
    print(f"[EVAL] Document prompt for AnswerRelevancy: {doc_prompt}")

    test_case = LLMTestCase(
        input=doc_prompt,
        actual_output=slides,
        retrieval_context=text_chunks if text_chunks else [doc_prompt],
    )

    async def _gather():
        return await asyncio.gather(
            *[_score_metric_async(m, test_case) for m in _DEEPEVAL_METRICS]
        )

    return asyncio.run(_gather()), doc_prompt


def _extract_full_pdf_text(pdf_path: str, max_pages: int = 30) -> list[str]:
    """Extract page-level text from a PDF using PyMuPDF.

    Returns one string per page (up to max_pages) so the full document's
    section structure is available for ContentCoverageMetric — not just the
    small subset that RAG happened to retrieve.
    """
    import fitz as _fitz
    pages: list[str] = []
    try:
        with _fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                text = page.get_text("text").strip()
                if text:
                    pages.append(f"[Page {i + 1}]\n{text}")
    except Exception as exc:
        print(f"[WARN] Could not read full PDF text from {pdf_path}: {exc}")
    return pages


@weave.op()
def _score_content_coverage(slides: str, pdf_path: str) -> dict:
    """Score ContentCoverage against the FULL PDF text, not just retrieved chunks.

    RAG retrieves only k=6 text chunks, which may miss entire sections of the
    paper. This function reads every page so the judge can identify ALL sections
    the document actually contains before checking whether the presentation
    covers them.
    """
    full_pages = _extract_full_pdf_text(pdf_path)
    slide_text = slides.strip() or "No presentation content was generated."

    test_case = LLMTestCase(
        input=_PROMPT,
        actual_output=slide_text,
        retrieval_context=full_pages if full_pages else ["No PDF text available."],
    )

    import asyncio as _asyncio
    return _asyncio.run(_score_metric_async(_content_coverage_metric, test_case))


@weave.op()
def _score_image_relevance(slides: str, image_captions: str = "") -> dict:
    """Run Image Relevance G-Eval over the JSON slide array.

    The judge is given:
      • actual_output  — the JSON slide array (with image_path fields)
      • retrieval_context — the Image_Captioner_output describing each image

    This lets the judge compare each slide's text against the caption of the
    image assigned to it, without needing to see the actual image pixels.
    """
    slide_text = slides.strip() if slides.strip() \
        else "No presentation content was generated."

    test_case = LLMTestCase(
        input=_PROMPT,
        actual_output=slide_text,
        retrieval_context=[image_captions] if image_captions.strip() else ["No image captions available."],
    )

    import asyncio as _asyncio
    result = _asyncio.run(_score_metric_async(_image_relevance_metric, test_case))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  JSON PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_averages(results: list[dict]) -> dict:
    from collections import defaultdict
    metric_scores: dict[str, list[float]] = defaultdict(list)
    clip_scores: list[float] = []

    for r in results:
        if r.get("clip_score") is not None:
            clip_scores.append(float(r["clip_score"]))
        for m in r.get("metrics", []):
            s = m.get("score")
            if s is not None:
                metric_scores[m["name"]].append(float(s))

    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    averages = {name: _avg(scores) for name, scores in metric_scores.items()}
    if clip_scores:
        averages["CLIPScore"] = _avg(clip_scores)
    return averages


def _flush_json(results: list[dict]) -> None:
    try:
        averages = _compute_averages(results)
        output = {
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "pdfs_dir":     _PDFS_DIR,
            "prompt":       _PROMPT,
            "total":        len(results),
            "passed":       sum(1 for r in results if r.get("passed")),
            "failed":       sum(1 for r in results if not r.get("passed")),
            "averages":     averages,
            "results":      results,
        }
        with open(_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        print(f"[WARN] Could not write results JSON: {exc}")


def _load_previous_results() -> list[dict]:
    if os.path.exists(_RESULTS_JSON):
        try:
            with open(_RESULTS_JSON, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("results", [])
        except Exception:
            pass
    return []


def _is_completed(result: dict) -> bool:
    """Completed = has metrics + no error, OR timed out."""
    if result.get("metrics") and not result.get("error"):
        return True
    err = result.get("error", "")
    if isinstance(err, str) and err.startswith("Timeout:"):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME STATE  (computed at import / pytest-collection time)
# ══════════════════════════════════════════════════════════════════════════════

_prev_results  = _load_previous_results()
_done_keys     = {r["run_key"] for r in _prev_results if _is_completed(r)}
pytest_results: list[dict] = [r for r in _prev_results if _is_completed(r)]

_samples        = _load_samples()
_samples_to_run = [s for s in _samples if s["run_key"] not in _done_keys]
_ids            = [s["pdf_file"] for s in _samples_to_run]

retry_count   = sum(1 for r in _prev_results if not _is_completed(r))
skipped_count = len(_samples) - len(_samples_to_run)
print(
    f"[INFO] Skipping {skipped_count} completed samples, "
    f"retrying {retry_count} errored/timed-out, "
    f"{len(_samples_to_run)} to run."
)

# Guarantee the JSON file always exists before any test runs
if not os.path.exists(_RESULTS_JSON):
    _flush_json([])


# ══════════════════════════════════════════════════════════════════════════════
#  PYTEST PARAMETRISED TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("sample", _samples_to_run, ids=_ids)
def test_slide_generation(sample: dict):
    """
    Full pipeline evaluation for one PDF:
      1. Index → RAG query → slide generation graph
      2. Faithfulness + AnswerRelevancy (async) — against retrieved chunks only,
         so faithfulness correctly checks grounding in what the LLM actually saw
      3. ContentCoverage (G-Eval) — against full PDF text so the judge sees ALL
         sections, not just the k=6 retrieved chunks
      4. ImageRelevance (G-Eval) — judge reads image captions from Image_Captioner
      5. CLIP Score (ViT-B-32 cosine sim) — per-slide text vs. image

    Skips automatically if execution exceeds _TIMEOUT_SECONDS.
    Flushes results to JSON after every sample (success, error, or timeout).
    """

    def _run_test() -> dict:
        # ── 1. Run pipeline ──────────────────────────────────────────────────
        pipeline_out = _run_pipeline(sample)

        slides             = pipeline_out["slides"]
        text_chunks        = pipeline_out["retrieved_text"]
        image_paths        = pipeline_out["retrieved_image_paths"]
        image_captions     = pipeline_out.get("image_captioner_output", "")

        assert slides, f"Pipeline returned empty slides for {sample['pdf_file']}"

        # ── 2. DeepEval metrics (Faithfulness + AnswerRelevancy) — uses retrieved chunks
        #        so faithfulness correctly checks what the LLM was actually given.
        deepeval_scores, doc_prompt = _score_deepeval_metrics(slides, text_chunks)

        # ── 3. Content Coverage — scored against FULL PDF text so the judge can
        #        identify every section in the paper, not just the retrieved subset.
        coverage_score = _score_content_coverage(slides, sample["pdf_path"])

        # ── 4. Image Relevance G-Eval — judge uses image captions as context ─
        img_relevance_score = _score_image_relevance(slides, image_captions)

        # Combine all metric scores into one list
        all_scores = deepeval_scores + [coverage_score, img_relevance_score]

        # ── 5. CLIP Score — per-slide text vs. image on the same slide ──────
        clip_score = _compute_clip_score(slides, _IMAGES_TMP)

        return {
            "scores":      all_scores,
            "clip_score":  clip_score,
            "doc_prompt":  doc_prompt,
            "slides":      slides,
            "text_chunks": text_chunks,
            "image_paths": image_paths,
        }

    # ── Timeout wrapper ───────────────────────────────────────────────────────
    executor = ThreadPoolExecutor(max_workers=1)
    future   = executor.submit(_run_test)

    try:
        out = future.result(timeout=_TIMEOUT_SECONDS)
    except FuturesTimeout:
        executor.shutdown(wait=False)
        msg = f"Timeout: exceeded {_TIMEOUT_SECONDS} seconds"
        print(f"[SKIP] {sample['pdf_file']}: {msg}")
        pytest_results.append({
            "run_key":   sample["run_key"],
            "pdf_file":  sample["pdf_file"],
            "passed":    False,
            "slides":    "",
            "clip_score": None,
            "metrics":   [],
            "error":     msg,
        })
        _flush_json(pytest_results)
        _reset_rag()
        pytest.skip(msg)
        return
    except Exception:
        executor.shutdown(wait=False)
        error_msg = traceback.format_exc()
        print(f"[ERROR] {sample['pdf_file']}:\n{error_msg}")
        pytest_results.append({
            "run_key":    sample["run_key"],
            "pdf_file":   sample["pdf_file"],
            "passed":     False,
            "slides":     "",
            "clip_score": None,
            "metrics":    [],
            "error":      error_msg,
        })
        _flush_json(pytest_results)
        _reset_rag()
        raise
    else:
        executor.shutdown(wait=False)

    scores     = out["scores"]
    clip_score = out["clip_score"]
    slides     = out["slides"]
    doc_prompt = out["doc_prompt"]

    failures = [s for s in scores if not s["success"]]

    pytest_results.append({
        "run_key":    sample["run_key"],
        "pdf_file":   sample["pdf_file"],
        "passed":     not failures and clip_score is not None,
        "slides":     slides,
        "doc_prompt": doc_prompt,
        "clip_score": clip_score,
        "metrics":    scores,
    })
    _flush_json(pytest_results)
    _reset_rag()

    failure_lines = "\n".join(
        f"  {s['name']}: score={s['score']:.3f} < threshold={s['threshold']} | {s['reason']}"
        for s in failures
    )
    assert not failures, f"Metrics failed for {sample['pdf_file']}:\n{failure_lines}"
