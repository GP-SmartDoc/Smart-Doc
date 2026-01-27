---
layout: cover
title: "MDocAgent: A Multi-Modal Multi-Agent Framework for Document Understanding"
---

# MDocAgent  
## A Multi-Modal Multi-Agent Framework for Document Understanding

---

## Document Metadata

Unfortunately, the title of the document and the authors’ names and affiliations are not provided.

---

## Summary

The document introduces **MDocAgent**, a multi-modal multi-agent framework designed to provide comprehensive and accurate answers for document understanding tasks.

The framework operates in **five stages** and employs **five specialized agents**.

### Agents
- General Agent  
- Critical Agent  
- Text Agent  
- Image Agent  
- Summarizing Agent  

### Pipeline Stages
1. **Document Processing** – Extracts text and images from PDF documents  
2. **Context Retrieval** – Applies text-based and image-based Retrieval-Augmented Generation (RAG)  
3. **Preliminary Analysis** – The general agent generates an initial answer, while the critical agent extracts key information  
4. **Specialized Analysis** – Text and image agents process modality-specific content  
5. **Answer Synthesis** – Integrates outputs from all agents into a final response  

The framework outputs:
- Critical textual information  
- Detailed descriptions of important visual content

---

layout: two-cols

::left::

## Key Points & Research Questions

The document explores the following research questions:

1. Does **MDocAgent** achieve higher document understanding accuracy compared to existing RAG-based approaches?
2. Does each agent contribute meaningfully to the final answer?
3. How does the multi-agent design improve document understanding?

The implementation uses **Llama-3.1-8B-Instruct** as the base model for the text agent.

::right::

## Conclusions

MDocAgent combines multiple agents and modalities to generate accurate and comprehensive answers.

Experiments evaluate:
- Overall framework performance  
- The contribution of each individual agent  

A complete academic citation cannot be provided due to missing document metadata.

---

## Image: Evaluation Form for Predicted Answer

**File:** `test1.pdf_diagram_13_1.png`

### Description

The image shows an **evaluation form** used to assess the correctness of a model’s predicted answer.

It contains:
- **Question**
- **Predicted Answer**
- **Ground Truth Answer**

### Evaluation Instructions
- Return `1` if the predicted answer is correct  
- Return `0` if the predicted answer is incorrect  

### Expected Output Format
```json
{"correctness": 1}
