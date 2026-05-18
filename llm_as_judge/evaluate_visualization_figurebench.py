"""
Figurebench Visualization Evaluation — G-Eval + Faithfulness
=============================================================
Reads every row from
  llm_as_judge/mermaid test/figurebench_visualizations.csv

The diagram class (flowchart, sequenceDiagram, stateDiagram, etc.) is
detected automatically from the first line of the ground-truth `diagram`
column and mapped to the appropriate DiagramType.

CSV columns
-----------
  message_content : PDF text provided by the user as source material
  diagram         : ground-truth Mermaid diagram (used to detect class)
  prompt          : the user's request to the visualization system

Three G-Eval metrics
--------------------
1. Syntax Validity       — ACTUAL_OUTPUT only
   Is the output a valid ```mermaid … ``` block with the correct type?

2. Prompt Alignment      — INPUT + ACTUAL_OUTPUT
   Does the generated diagram correctly address the user's prompt?

3. Structural Correctness — ACTUAL_OUTPUT only
   Does the diagram follow correct structural conventions for its type?

4. Faithfulness          — ACTUAL_OUTPUT + RETRIEVAL_CONTEXT
   Score = |verified claims| / |total claims|  (|V| / |C|)

Results
-------
  llm_as_judge/visualization_eval_results_figurebench.json

Run (from project root)
-----------------------
    python llm_as_judge/evaluate_visualization_figurebench.py

Via pytest:
    pytest llm_as_judge/evaluate_visualization_figurebench.py -v
    pytest llm_as_judge/evaluate_visualization_figurebench.py -v -k flowchart
"""

import asyncio

import os
import sys
import csv
import json
import re
import tempfile
import uuid
from datetime import datetime

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import chromadb
import weave
import pytest
from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.graphs.visualization_graph import generate_visualization
from src.states.visualization_state import DiagramType
from src.vector_store.RAG import RAGEngine

# ── Weave tracing ─────────────────────────────────────────────────────────────
weave.init("gamer7dragon817-ain-shams-university/slide-generator-demo")


# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════

_HERE       = os.path.dirname(os.path.abspath(__file__))
_CSV_PATH   = os.path.join(_HERE, "mermaid test", "figurebench_visualizations.csv")
_RESULTS_JSON = os.path.join(_HERE, "visualization_eval_results_figurebench.json")


# ══════════════════════════════════════════════════════════════════════════════
#  RAG ENGINE  (shared ephemeral instance — models load once)
# ══════════════════════════════════════════════════════════════════════════════

# EphemeralClient so test data never touches the persistent chroma_db.
# RAGEngine is created once so YOLO / embedder models load only once.
_rag_client = chromadb.EphemeralClient()
_rag        = RAGEngine(_rag_client)


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGRAM CLASS → DiagramType MAPPING
# ══════════════════════════════════════════════════════════════════════════════

_CLASS_TO_TYPE: dict[str, DiagramType] = {
    "flowchart":       DiagramType.FLOWCHART,
    "graph":           DiagramType.FLOWCHART,   # legacy 'graph TD' syntax
    "sequenceDiagram": DiagramType.SEQUENCE,
    "stateDiagram":    DiagramType.STATE,
    "classDiagram":    DiagramType.CLASS,
    "erDiagram":       DiagramType.ER,
    "pie":             DiagramType.PIE,
    "mindmap":         DiagramType.MINDMAP,
}


# ══════════════════════════════════════════════════════════════════════════════
#  JUDGE MODEL  (W&B Inference — same as evaluate.py)
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
#  G-EVAL METRICS
# ══════════════════════════════════════════════════════════════════════════════

# 1. Syntax Validity ──────────────────────────────────────────────────────────
_syntax_metric = GEval(
    name="SyntaxValidityMetric",
    criteria=(
        "Evaluate whether the actual output is a syntactically valid Mermaid diagram.\n"
        "Requirements:\n"
        "  • The diagram is wrapped in a ```mermaid … ``` fenced code block.\n"
        "  • The first line inside the block declares the correct diagram type "
        "    (e.g. 'flowchart TD', 'sequenceDiagram', 'stateDiagram-v2', "
        "    'classDiagram', 'erDiagram', 'pie', 'mindmap').\n"
        "  • Nodes, edges, and keywords use the correct operators for that type:\n"
        "    flowchart → '-->', sequence → '->>', state → '-->', ER → '||--'.\n"
        "Score 1.0 when all requirements are satisfied.\n"
        "Score 0.5 when the diagram type is correct and structure is mostly valid "
        "    but has minor syntax issues (inconsistent spacing, missing labels).\n"
        "Score 0.0 when the output is not a valid Mermaid block or uses wrong syntax."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=judge,
)

# 2. Prompt Alignment ─────────────────────────────────────────────────────────
_alignment_metric = GEval(
    name="PromptAlignmentMetric",
    criteria=(
        "Evaluate whether the generated diagram correctly represents what the "
        "user asked for in the input prompt.\n"
        "Evaluation focus:\n"
        "  • Does the diagram address the domain / scenario described in the prompt?\n"
        "  • Is the diagram type (flowchart, sequence, class, etc.) appropriate "
        "    for the request?\n"
        "  • Are the main concepts and entities from the prompt present in the diagram?\n"
        "Score 1.0 when the diagram is fully aligned with the prompt's intent.\n"
        "Score 0.5 when the diagram is on the right topic but misses some concepts "
        "    or uses a slightly wrong diagram type.\n"
        "Score 0.0 when the diagram addresses a different topic or is completely "
        "    misaligned with the user's request."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
    model=judge,
)

# 3. Structural Correctness ───────────────────────────────────────────────────
_structural_metric = GEval(
    name="StructuralCorrectnessMetric",
    criteria=(
        "Evaluate whether the diagram has a logically correct structure "
        "appropriate to its diagram type.\n"
        "Type-specific rules:\n"
        "  • Flowchart: flow direction is logical (start → process → end), "
        "    no unexpected backward arrows, decision nodes branch properly.\n"
        "  • Sequence: messages flow between named participants in a realistic "
        "    order; responses follow requests.\n"
        "  • State: transitions connect reachable states; no transitions out of "
        "    terminal [*]; initial state is clearly defined.\n"
        "  • Class: inheritance, association, and composition relationships are "
        "    semantically correct between classes.\n"
        "  • ER: entity relationships and cardinalities are realistic and consistent.\n"
        "  • Pie: all segments have distinct labels; values are non-negative.\n"
        "  • Mindmap: hierarchy is logical with a clear root and meaningful sub-topics.\n"
        "Score 1.0 when the diagram follows correct structural conventions.\n"
        "Score 0.5 when structure is mostly correct but has minor logical issues "
        "    (e.g. one redundant arrow, slightly odd hierarchy).\n"
        "Score 0.0 when the diagram has significant structural errors or "
        "    illogical flow that makes it misleading or unusable."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=judge,
)

_ALL_METRICS = [
    _syntax_metric,
    _alignment_metric,
    _structural_metric,
    FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True),
]


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE HELPER
# ══════════════════════════════════════════════════════════════════════════════

@weave.op()
def _run_full_pipeline(sample: dict) -> tuple[str, list[str]]:
    """
    Full RAG → Visualization pipeline for one sample:

    1. Write message_content to a temporary .txt file.
    2. Ingest it into the shared RAGEngine via add_txt().
    3. Query the RAG with the user's prompt to retrieve relevant chunks.
    4. Build an enriched description (context + prompt) for the LLM.
    5. Call generate_visualization() to produce the Mermaid diagram.

    Returns:
      actual_output   : the generated Mermaid diagram string
      retrieved_chunks: text chunks returned by RAG (used as retrieval_context
                        in the LLMTestCase for the FaithfulnessMetric)
    """
    # ─ Step 1 & 2: write content to temp file and ingest into RAG ───────────
    tmp_name = f"figurebench_{uuid.uuid4().hex}.txt"
    tmp_path = os.path.join(tempfile.gettempdir(), tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(sample["message_content"])
        _rag.add_txt(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # ─ Step 3: retrieve relevant chunks using the user's prompt ────────────
    rag_result      = _rag.query(sample["prompt"], k_text=6)
    retrieved_chunks: list[str] = rag_result.get("text", [])

    # ─ Step 3b: clean up — delete this sample's chunks from ChromaDB so the
    #   collection doesn't grow with every sample (keeps queries fast and
    #   prevents chunks from one sample polluting the next).
    chunk_basename = os.path.basename(tmp_name)
    for col_name in ("english_text", "arabic_text"):
        try:
            col = _rag_client.get_collection(col_name)
            existing = col.get()
            ids_to_delete = [
                id_ for id_ in existing["ids"]
                if id_.startswith(chunk_basename)
            ]
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
        except Exception:
            pass  # collection may not exist if no chunks were routed there

    # ─ Step 4: build enriched description for the visualization LLM ───────
    context_block = "\n\n".join(retrieved_chunks) if retrieved_chunks else ""
    if context_block:
        enriched_description = (
            f"Context from document:\n{context_block}\n\n"
            f"User request: {sample['prompt']}"
        )
    else:
        enriched_description = sample["prompt"]

    # ─ Step 5: generate the diagram ───────────────────────────────────────
    actual_output = generate_visualization(
        type=sample["diagram_type"],
        description=enriched_description,
    )
    return actual_output, retrieved_chunks


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _strip_fence(text: str) -> str:
    """Remove ```mermaid … ``` or ``` … ``` wrapper if present."""
    text = text.strip()
    text = re.sub(r"^```(?:mermaid)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _detect_class(diagram: str) -> str | None:
    """
    Detect the diagram class from the first content line of the diagram.
    Returns the matching key from _CLASS_TO_TYPE, or None if unrecognised.
    """
    first_line = _strip_fence(diagram).splitlines()[0].strip() if diagram else ""
    # Match longest key first to avoid 'graph' shadowing 'graph TD' etc.
    for keyword in sorted(_CLASS_TO_TYPE.keys(), key=len, reverse=True):
        if first_line.lower().startswith(keyword.lower()):
            return keyword
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  CSV LOADER  — all diagram types
# ══════════════════════════════════════════════════════════════════════════════

def _load_samples() -> list[dict]:
    """
    Read figurebench_visualizations.csv and return all recognised rows.
    The diagram class and DiagramType are detected from the `diagram` column.

    Each returned dict has:
      prompt          : str        — user's request
      ground_truth    : str        — ground-truth Mermaid diagram
      message_content : str        — PDF text provided by the user
      diagram_class   : str        — detected class (e.g. 'flowchart')
      diagram_type    : DiagramType
    """
    samples: list[dict] = []
    skipped = 0

    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            diagram = row.get("diagram", "").strip()
            cls_key = _detect_class(diagram)
            if cls_key is None:
                print(f"[WARN] Unrecognised diagram type — skipping row: "
                      f"{diagram[:60]!r}")
                skipped += 1
                continue

            samples.append({
                "prompt":          row["prompt"].strip(),
                "ground_truth":    diagram,
                "message_content": row["message_content"].strip(),
                "diagram_class":   cls_key,
                "diagram_type":    _CLASS_TO_TYPE[cls_key],
            })

    from collections import Counter
    class_counts = Counter(s["diagram_class"] for s in samples)
    print(f"[INFO] Loaded {len(samples)} samples "
          f"(skipped {skipped} unrecognised): "
          + ", ".join(f"{k}={v}" for k, v in sorted(class_counts.items())))
    return samples


# ══════════════════════════════════════════════════════════════════════════════
#  SCORE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _score_metric(metric, test_case: LLMTestCase) -> dict:
    metric.measure(test_case)
    return {
        "name":      getattr(metric, "name", type(metric).__name__),
        "score":     metric.score,
        "threshold": metric.threshold,
        "success":   metric.score >= metric.threshold,
        "reason":    getattr(metric, "reason", None),
    }


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
def _score_all_metrics(test_case: LLMTestCase) -> list[dict]:
    """Run all metrics concurrently and return results in _ALL_METRICS order."""
    async def _gather():
        return await asyncio.gather(
            *[_score_metric_async(m, test_case) for m in _ALL_METRICS]
        )
    return asyncio.run(_gather())


# ══════════════════════════════════════════════════════════════════════════════
#  AVERAGES HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _compute_averages(results: list[dict]) -> dict:
    from collections import defaultdict
    overall: dict[str, list[float]] = defaultdict(list)
    by_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        cls = r.get("diagram_class", "unknown")
        for m in r.get("metrics", []):
            s = m.get("score")
            if s is not None:
                overall[m["name"]].append(float(s))
                by_class[cls][m["name"]].append(float(s))

    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "overall": {name: _avg(scores) for name, scores in overall.items()},
        "by_class": {
            cls: {name: _avg(scores) for name, scores in metrics.items()}
            for cls, metrics in sorted(by_class.items())
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FLUSH JSON
# ══════════════════════════════════════════════════════════════════════════════

def _flush_json(results: list[dict]) -> None:
    """Write current results to disk. Wrapped in try/except so a write
    failure never aborts the evaluation run."""
    try:
        averages = _compute_averages(results)
        output = {
            "timestamp":        datetime.now().isoformat(timespec="seconds"),
            "csv_source":       _CSV_PATH,
            "total":            len(results),
            "passed":           sum(1 for r in results if r["passed"]),
            "failed":           sum(1 for r in results if not r["passed"]),
            "overall_averages": averages["overall"],
            "class_averages":   averages["by_class"],
            "results":          results,
        }
        with open(_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not write results JSON: {exc}")


def _write_skeleton() -> None:
    """Write an empty result file at startup so the file always exists,
    even if the run crashes before the first sample completes."""
    if not os.path.exists(_RESULTS_JSON):
        _flush_json([])


# ══════════════════════════════════════════════════════════════════════════════
#  PYTEST PARAMETRISED TESTS
#
#  Run all:    pytest llm_as_judge/evaluate_visualization_figurebench.py -v
#  Auto-skips completed samples. Re-runs samples that errored/timed-out.
# ══════════════════════════════════════════════════════════════════════════════

def _load_previous_results() -> list[dict]:
    """Load already-completed results from the JSON file (if it exists)."""
    if os.path.exists(_RESULTS_JSON):
        try:
            with open(_RESULTS_JSON, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("results", [])
        except Exception:
            pass
    return []


def _is_completed(result: dict) -> bool:
    """A result counts as completed if it has metric scores OR it timed out (skip both)."""
    if result.get("metrics") and not result.get("error"):
        return True
    err = result.get("error", "")
    if isinstance(err, str) and err.startswith("Timeout:"):
        return True
    return False


_prev_results = _load_previous_results()
# Prompts that completed successfully or timed out — skip these
_done_prompts = {r["prompt"] for r in _prev_results if _is_completed(r)}
# Keep previous completed + timeout results; drop other errors so they get re-run
pytest_results: list[dict] = [r for r in _prev_results if _is_completed(r)]

_samples        = _load_samples()
_samples_to_run = [s for s in _samples if s["prompt"] not in _done_prompts]
_ids            = [f"{s['diagram_class']}/{s['prompt'][:55]}" for s in _samples_to_run]

retry_count  = sum(1 for r in _prev_results if not _is_completed(r))
skipped_count = len(_samples) - len(_samples_to_run)
print(f"[INFO] Skipping {skipped_count} completed samples, "
      f"retrying {retry_count} errored/timed-out, "
      f"{len(_samples_to_run)} to run.")

_write_skeleton()   # guarantee the JSON file exists before any test runs


@pytest.mark.parametrize("sample", _samples_to_run, ids=_ids)
def test_visualization_figurebench(sample: dict):
    """
    For each row in figurebench_visualizations.csv runs the FULL pipeline:
      1. Ingest message_content into RAG via add_txt().
      2. Query RAG with the user's prompt to retrieve relevant chunks.
      3. Pass retrieved context + prompt to generate_visualization().
      4. Evaluate the output with 4 G-Eval metrics.

    Skipped (pytest.skip) if the entire test exceeds 60 seconds.
    """
    import traceback
    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    _TIMEOUT_SECONDS = 180

    def _run_test():
        actual, retrieved_chunks = _run_full_pipeline(sample)
        assert actual, f"Pipeline returned empty output for: {sample['prompt'][:80]}"

        test_case = LLMTestCase(
            input=sample["prompt"],
            actual_output=actual,
            retrieval_context=retrieved_chunks if retrieved_chunks else [sample["message_content"]],
        )

        scores   = _score_all_metrics(test_case)
        return actual, retrieved_chunks, scores

    try:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_run_test)
        try:
            actual, retrieved_chunks, scores = future.result(timeout=_TIMEOUT_SECONDS)
        except FuturesTimeout:
            # shutdown(wait=False) abandons the thread immediately instead of blocking
            executor.shutdown(wait=False)
            print(f"[SKIP] Test exceeded {_TIMEOUT_SECONDS}s: {sample['prompt'][:80]}")
            pytest_results.append({
                "prompt":           sample["prompt"],
                "diagram_class":    sample["diagram_class"],
                "passed":           False,
                "actual_output":    "",
                "ground_truth":     sample["ground_truth"],
                "retrieved_chunks": [],
                "error":            f"Timeout: exceeded {_TIMEOUT_SECONDS} seconds",
                "metrics":          [],
            })
            _flush_json(pytest_results)
            pytest.skip(f"Exceeded {_TIMEOUT_SECONDS}s time limit")
        except Exception:
            executor.shutdown(wait=False)
            error_msg = traceback.format_exc()
            print(f"[ERROR] Test failed for: {sample['prompt'][:80]}\n{error_msg}")
            pytest_results.append({
                "prompt":           sample["prompt"],
                "diagram_class":    sample["diagram_class"],
                "passed":           False,
                "actual_output":    "",
                "ground_truth":     sample["ground_truth"],
                "retrieved_chunks": [],
                "error":            error_msg,
                "metrics":          [],
            })
            _flush_json(pytest_results)
            raise
        else:
            executor.shutdown(wait=False)

        failures = [s for s in scores if not s["success"]]
        pytest_results.append({
            "prompt":            sample["prompt"],
            "diagram_class":     sample["diagram_class"],
            "passed":            not failures,
            "actual_output":     actual,
            "ground_truth":      sample["ground_truth"],
            "retrieved_chunks":  retrieved_chunks,
            "metrics":           scores,
        })

        assert not failures, "G-Eval metrics failed:\n" + "\n".join(
            f"  {s['name']}: score={s['score']:.3f} < threshold={s['threshold']}"
            f"  | {s['reason']}"
            for s in failures
        )
    finally:
        _flush_json(pytest_results)


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECT RUN  (python llm_as_judge/evaluate_visualization_figurebench.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_all() -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print("  Smart-Doc — Figurebench Visualization G-Eval")
    print(sep)

    samples  = _load_samples()
    results  = []
    total    = len(samples)
    passed_n = 0

    for idx, sample in enumerate(samples, 1):
        print(f"\n[{idx}/{total}]  [{sample['diagram_class']}]  "
              f"{sample['prompt'][:70]} …")

        import traceback as _tb
        try:
            actual, retrieved_chunks = _run_full_pipeline(sample)
        except Exception:
            error_msg = _tb.format_exc()
            print(f"  [ERROR] Pipeline crashed:\n{error_msg}")
            results.append({
                "prompt":           sample["prompt"],
                "diagram_class":    sample["diagram_class"],
                "passed":           False,
                "actual_output":    "",
                "ground_truth":     sample["ground_truth"],
                "retrieved_chunks": [],
                "error":            error_msg,
                "metrics":          [],
            })
            _flush_json(results)
            continue

        if not actual:
            print("  [WARN] Pipeline returned empty output — skipping.")
            results.append({
                "prompt":           sample["prompt"],
                "diagram_class":    sample["diagram_class"],
                "passed":           False,
                "actual_output":    "",
                "ground_truth":     sample["ground_truth"],
                "retrieved_chunks": retrieved_chunks,
                "metrics": [
                    {"name": m.name, "score": 0.0,
                     "threshold": m.threshold, "success": False,
                     "reason": "Pipeline returned empty output"}
                    for m in _ALL_METRICS
                ],
            })
            _flush_json(results)
            continue

        test_case = LLMTestCase(
            input=sample["prompt"],
            actual_output=actual,
            retrieval_context=retrieved_chunks if retrieved_chunks else [sample["message_content"]],
        )

        scores    = _score_all_metrics(test_case)
        passed    = all(s["success"] for s in scores)
        passed_n += int(passed)

        status = "PASS" if passed else "FAIL"
        for s in scores:
            icon = "✓" if s["success"] else "✗"
            print(f"  {icon} [{s['name']}]  score={s['score']:.3f}  "
                  f"threshold={s['threshold']}  {status}")
            if not s["success"]:
                print(f"    Reason: {s['reason']}")

        results.append({
            "prompt":           sample["prompt"],
            "diagram_class":    sample["diagram_class"],
            "passed":           passed,
            "actual_output":    actual,
            "ground_truth":     sample["ground_truth"],
            "retrieved_chunks": retrieved_chunks,
            "metrics":          scores,
        })
        _flush_json(results)

    # ── Summary ────────────────────────────────────────────────────────────
    averages = _compute_averages(results)

    print(f"\n{sep}")
    print(f"  Results : {passed_n}/{total} passed")
    print(f"\n  Overall averages:")
    for metric_name, avg in averages["overall"].items():
        print(f"    {metric_name}: {avg:.4f}")
    print(f"\n  Per diagram-class averages:")
    for cls, metrics in averages["by_class"].items():
        print(f"    [{cls}]")
        for metric_name, avg in metrics.items():
            print(f"      {metric_name}: {avg:.4f}")
    # ── Guaranteed final save ──────────────────────────────────────────────
    _flush_json(results)
    print(f"\n  Saved : {_RESULTS_JSON}")
    print(sep)


if __name__ == "__main__":
    run_all()
