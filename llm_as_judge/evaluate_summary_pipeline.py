"""
Full Pipeline Summary Evaluation — RAG + SummarizationModule
=============================================================
PDFs  :  llm_as_judge/summary_test/temp/test{N}.pdf  (30 PDFs)
Modes :  deepdive, overview, snapshot  (3 per PDF → 90 samples total)
Prompt:  "summarize all the document"

Pipeline per (PDF, mode)
-------------------------
  1. Lazy PDF indexing into a shared ephemeral RAG (once per PDF).
  2. Run SummarizationModule.invoke(prompt, document=pdf_file, summary_mode=mode)
     → produces the actual generated summary.
  3. Query RAG separately (mirrors SummarizationModule's internal search query)
     → retrieval_context for metric evaluation.
  4. Evaluate with three DeepEval metrics (async, concurrent):
       • FaithfulnessMetric        — summary grounded in retrieved context
       • AnswerRelevancyMetric     — summary addresses the prompt
       • ContextualRelevancyMetric — retrieved chunks are relevant to the prompt

Output
------
    llm_as_judge/summary_pipeline_eval_results.json   (flushed after EVERY sample)

Averages are computed overall AND per mode (deepdive / overview / snapshot).

Resume / Retry
--------------
    Completed samples (metrics present, no error) and timed-out samples are skipped.
    Errored samples are retried on the next run.

Run (from project root)
-----------------------
    pytest llm_as_judge/evaluate_summary_pipeline.py -v
"""

import asyncio
import os
import sys
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from collections import defaultdict

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import chromadb
import weave
import pytest
from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase

from src.vector_store.RAG import RAGEngine
from src.graphs.summary_graph import SummarizationModule

# ── Weave tracing ─────────────────────────────────────────────────────────────
weave.init("gamer7dragon817-ain-shams-university/slide-generator-demo")


# ══════════════════════════════════════════════════════════════════════════════
#  PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PDFS_DIR     = os.path.join(_HERE, "summary_test", "temp")
_RESULTS_JSON = os.path.join(_HERE, "summary_pipeline_eval_results.json")

_PROMPT  = "summarize all the document"
_MODES   = ["overview", "deepdive", "snapshot"]

# SummarizationModule intercepts short summary queries (≤5 words) and replaces
# the vector-store search with a richer query internally.  Mirror that here so
# the retrieval_context we pass to the judge matches what the module actually used.
_SEARCH_QUERY = "abstract introduction main contribution methodology conclusion"

# k_text values per mode (match the module's internal defaults)
_MODE_K: dict[str, int] = {
    "overview": 6,
    "deepdive": 6,
    "snapshot": 6,
}

_TIMEOUT_SECONDS = 600   # per (PDF, mode) — indexing + 6 LLM calls + 3 metrics


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
    threshold=0.7,
    model=judge,
    include_reason=True,
)

_contextual_relevancy_metric = ContextualRelevancyMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
)

_ALL_METRICS = [
    _faithfulness_metric,
    _answer_relevancy_metric,
    _contextual_relevancy_metric,
]


# ══════════════════════════════════════════════════════════════════════════════
#  RAG + SUMMARIZATION ENGINE  (shared ephemeral instance)
# ══════════════════════════════════════════════════════════════════════════════

_rag_client   = chromadb.EphemeralClient()
_rag          = RAGEngine(_rag_client)
_summarizer   = SummarizationModule(_rag)
_indexed_pdfs: set[str] = set()   # lazy — index each PDF only once


# ══════════════════════════════════════════════════════════════════════════════
#  SAMPLE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_samples() -> list[dict]:
    """
    Discover all test{N}.pdf files and expand to (pdf × mode) pairs.
    Returns samples sorted by pdf_file then mode.
    """
    samples = []
    for fname in sorted(os.listdir(_PDFS_DIR)):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(_PDFS_DIR, fname)
        for mode in _MODES:
            samples.append({
                "pdf_file": fname,
                "pdf_path": pdf_path,
                "mode":     mode,
                "run_key":  f"{fname}__{mode}",
            })
    print(f"[INFO] {len(samples)} samples ({len(samples) // len(_MODES)} PDFs × {len(_MODES)} modes)")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

@weave.op()
def _run_pipeline(sample: dict) -> tuple[str, list[str]]:
    """
    Lazy index PDF → run SummarizationModule → query RAG for retrieval_context.

    Returns:
      summary_text    : str       — generated summary (actual_output for judge)
      retrieved_chunks: list[str] — chunks from RAG (retrieval_context for judge)
    """
    pdf_file = sample["pdf_file"]
    pdf_path = sample["pdf_path"]
    mode     = sample["mode"]

    # ── Lazy PDF indexing (shared across all 3 modes for the same PDF) ────────
    if pdf_file not in _indexed_pdfs:
        print(f"[INDEX] Indexing {pdf_file} …")
        _rag.add_pdf(pdf_path)
        _indexed_pdfs.add(pdf_file)

    # ── Run full summarization pipeline ───────────────────────────────────────
    print(f"[SUMMARIZE] {pdf_file} / {mode} …")
    final_summary: dict = _summarizer.invoke(
        _PROMPT,
        document=pdf_file,
        summary_mode=mode,
    )

    # Extract plain-text summary — try "Answer" key first, then dump full dict
    if isinstance(final_summary, dict):
        summary_text = final_summary.get(
            "Answer",
            final_summary.get(
                "answer",
                json.dumps(final_summary, ensure_ascii=False)
            )
        )
    else:
        summary_text = str(final_summary)

    # ── Retrieve context separately for the judge ─────────────────────────────
    # Use the same intercepted search_query that SummarizationModule uses
    # internally for short summary-style prompts.
    k = _MODE_K[mode]
    retrieved = _rag.query(_SEARCH_QUERY, k_text=k, document=pdf_file)
    retrieved_chunks: list[str] = retrieved.get("text", [])

    # Fall back: if the intercepted query returned nothing, try the raw prompt
    if not retrieved_chunks:
        retrieved = _rag.query(_PROMPT, k_text=k, document=pdf_file)
        retrieved_chunks = retrieved.get("text", [])

    return summary_text, retrieved_chunks


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


@weave.op()
def _score_all_metrics(
    summary_text: str,
    retrieved_chunks: list[str],
) -> list[dict]:
    """Run all 3 metrics concurrently and return results."""
    test_case = LLMTestCase(
        input=_PROMPT,
        actual_output=summary_text,
        retrieval_context=retrieved_chunks if retrieved_chunks else [_PROMPT],
    )

    async def _gather():
        return await asyncio.gather(
            *[_score_metric_async(m, test_case) for m in _ALL_METRICS]
        )

    return asyncio.run(_gather())


# ══════════════════════════════════════════════════════════════════════════════
#  JSON PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_averages(results: list[dict]) -> dict:
    """Overall + per-mode averages for every metric."""
    overall: dict[str, list[float]] = defaultdict(list)
    by_mode: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        mode = r.get("mode", "unknown")
        for m in r.get("metrics", []):
            s = m.get("score")
            if s is not None:
                overall[m["name"]].append(float(s))
                by_mode[mode][m["name"]].append(float(s))

    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "overall": {name: _avg(scores) for name, scores in overall.items()},
        "by_mode": {
            mode: {name: _avg(scores) for name, scores in metrics.items()}
            for mode, metrics in sorted(by_mode.items())
        },
    }


def _flush_json(results: list[dict]) -> None:
    try:
        averages = _compute_averages(results)
        output = {
            "timestamp":   datetime.now().isoformat(timespec="seconds"),
            "pdfs_dir":    _PDFS_DIR,
            "prompt":      _PROMPT,
            "modes":       _MODES,
            "total":       len(results),
            "passed":      sum(1 for r in results if r.get("passed")),
            "failed":      sum(1 for r in results if not r.get("passed")),
            "averages":    averages,
            "results":     results,
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
    """Completed = has metrics + no error. Timeout samples are also skipped."""
    if result.get("metrics") and not result.get("error"):
        return True
    err = result.get("error", "")
    if isinstance(err, str) and err.startswith("Timeout:"):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  RESUME STATE  (computed at import / pytest-collection time)
# ══════════════════════════════════════════════════════════════════════════════

_prev_results   = _load_previous_results()
_done_keys      = {r["run_key"] for r in _prev_results if _is_completed(r)}
pytest_results: list[dict] = [r for r in _prev_results if _is_completed(r)]

_samples        = _load_samples()
_samples_to_run = [s for s in _samples if s["run_key"] not in _done_keys]
_ids            = [f"{s['pdf_file']}/{s['mode']}" for s in _samples_to_run]

retry_count   = sum(1 for r in _prev_results if not _is_completed(r))
skipped_count = len(_samples) - len(_samples_to_run)
print(
    f"[INFO] Skipping {skipped_count} completed samples, "
    f"retrying {retry_count} errored/timed-out, "
    f"{len(_samples_to_run)} to run."
)

if not os.path.exists(_RESULTS_JSON):
    _flush_json([])


# ══════════════════════════════════════════════════════════════════════════════
#  PYTEST PARAMETRISED TESTS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("sample", _samples_to_run, ids=_ids)
def test_summary_pipeline(sample: dict):
    """
    Full pipeline evaluation for one (PDF, mode) pair:
      1. Lazy PDF indexing → SummarizationModule.invoke()
      2. Separate RAG query → retrieval_context for the judge
      3. FaithfulnessMetric + AnswerRelevancyMetric + ContextualRelevancyMetric
         (all three run concurrently via asyncio.gather)

    Skips automatically if execution exceeds _TIMEOUT_SECONDS.
    Flushes JSON after every sample (success, error, or timeout).
    """

    def _run_test() -> tuple[str, list[str], list[dict]]:
        summary_text, retrieved_chunks = _run_pipeline(sample)
        assert summary_text, (
            f"SummarizationModule returned empty output for "
            f"{sample['pdf_file']} / {sample['mode']}"
        )
        scores = _score_all_metrics(summary_text, retrieved_chunks)
        return summary_text, retrieved_chunks, scores

    # ── Timeout wrapper ───────────────────────────────────────────────────────
    executor = ThreadPoolExecutor(max_workers=1)
    future   = executor.submit(_run_test)

    try:
        summary_text, retrieved_chunks, scores = future.result(
            timeout=_TIMEOUT_SECONDS
        )
    except FuturesTimeout:
        executor.shutdown(wait=False)
        msg = f"Timeout: exceeded {_TIMEOUT_SECONDS} seconds"
        print(f"[SKIP] {sample['pdf_file']}/{sample['mode']}: {msg}")
        pytest_results.append({
            "run_key":           sample["run_key"],
            "pdf_file":          sample["pdf_file"],
            "mode":              sample["mode"],
            "passed":            False,
            "prompt":            _PROMPT,
            "summary":           "",
            "retrieval_context": [],
            "metrics":           [],
            "error":             msg,
        })
        _flush_json(pytest_results)
        pytest.skip(msg)
        return
    except Exception:
        executor.shutdown(wait=False)
        error_msg = traceback.format_exc()
        print(f"[ERROR] {sample['pdf_file']}/{sample['mode']}:\n{error_msg}")
        pytest_results.append({
            "run_key":           sample["run_key"],
            "pdf_file":          sample["pdf_file"],
            "mode":              sample["mode"],
            "passed":            False,
            "prompt":            _PROMPT,
            "summary":           "",
            "retrieval_context": [],
            "metrics":           [],
            "error":             error_msg,
        })
        _flush_json(pytest_results)
        raise
    else:
        executor.shutdown(wait=False)

    failures = [s for s in scores if not s["success"]]

    pytest_results.append({
        "run_key":           sample["run_key"],
        "pdf_file":          sample["pdf_file"],
        "mode":              sample["mode"],
        "passed":            not failures,
        "prompt":            _PROMPT,
        "summary":           summary_text,
        "retrieval_context": retrieved_chunks,
        "metrics":           scores,
    })
    _flush_json(pytest_results)

    assert not failures, (
        f"Metrics failed for {sample['pdf_file']}/{sample['mode']}:\n"
        + "\n".join(
            f"  {s['name']}: score={s['score']:.3f} < "
            f"threshold={s['threshold']} | {s['reason']}"
            for s in failures
        )
    )
