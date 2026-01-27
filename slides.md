---
layout: cover
title: MDocAgent Framework
---

# MDocAgent Framework
## Overview

The MDocAgent framework is a multi-modal, multi-agent framework designed to process documents and provide comprehensive answers to questions. It operates in five stages:

- **Document Processing**: Documents are processed using PDF tools to extract text and images.
- **Context Retrieval**: Text-based and image-based RAG retrieves the top-k relevant segments and image pages.
- **Preliminary Answer Generation**: The general agent provides a preliminary answer, and the critical agent extracts critical information from both modalities.
- **Specialized Analysis**: Specialized agents process the retrieved context.
- **Answer Synthesis**: The summarizing agent integrates the outputs from the preceding stages to produce a comprehensive and accurate answer.

---
layout: two-cols
---

::left::
## Key Components and Functions

* The general agent generates a preliminary answer by combining information from both textual and visual contexts.
* The critical agent extracts critical information from both modalities.
* The summarizing agent receives answers from the general agent, text agent, and image agent to produce a comprehensive answer.

::right::
## Research Questions and Experiment Setup

The document also mentions three research questions:
1. Does MDocAgent improve document understanding accuracy compared to existing RAG-based approaches?
2. Does each agent in the framework play a meaningful role?
3. How does MDocAgent enhance the model's understanding of documents?

The experiment setup includes implementing MDocAgent with five agents and using Llama-3.1-8B-Instruct as the base model for the text agent.

---
layout: cover
---

## MDocAgent Framework Diagram
![MDocAgent Framework Diagram](images/test1.pdf_diagram_13_1.png)

## Detailed Explanation
The image shows a diagram representing the MDocAgent framework, which is a multi-modal, multi-agent framework designed to operate in five stages. These stages include:
1. **Document Processing**: Utilizing PDF tools to extract text and images from documents.
2. **Text-based and Image-based RAG Retrieval**: Retrieving the top-k relevant segments and image pages.
3. **Preliminary Answer Generation and Critical Information Extraction**: A general agent provides a preliminary answer, and a critical agent extracts critical information from both modalities.
4. **Specialized Agent Processing**: Specialized agents process the retrieved context.
5. **Answer Synthesis**: Integrating diverse outputs to produce a comprehensive and accurate answer.

The diagram illustrates the structured approach of the MDocAgent framework in understanding documents through multiple agents and modalities.

---
layout: cover
---

## Conclusion

The MDocAgent framework presents a novel multi-modal multi-agent approach for document understanding, integrating the strengths of different agents to provide a comprehensive answer. The framework's architecture and experiment setup are designed to evaluate its effectiveness and the role of each agent.

---
layout: cover
---

## Summary

- The MDocAgent framework operates in five stages to process documents and provide comprehensive answers.
- The framework consists of five agents: general agent, critical agent, text agent, image agent, and summarizing agent.
- The experiment setup includes implementing MDocAgent with five agents and using Llama-3.1-8B-Instruct as the base model for the text agent.
