---
name: paper-review
description: Systematic review of an academic paper — methodology, experiments, contributions, and limitations. Use when the user asks to review, critique, or evaluate a paper.
---

# Paper Review

## Workflow

1. **Metadata**: Use `fetch_content(paper_name)` to get title, authors, abstract, and section list.
2. **Deep Read**: Use `fetch_content(paper_name, section)` to read each major section:
   - Introduction → understand the problem statement
   - Methodology → evaluate soundness, baselines, ablation studies
   - Experiments → assess metrics, significance, reproducibility
   - Conclusion → note limitations and future work
3. **Literature Check**: If the user asks about novelty, use `arxiv__search_papers` to find related work on arXiv.

## Output Format

Output in structured markdown:

```
## Paper Review: [Title]

**Authors**: ...
**Venue/Year**: ...

### 1. Problem & Contributions
- ...

### 2. Methodology
- Approach: ...
- Baselines: ...
- Ablation: ...

### 3. Experimental Results
- Key metrics: ...
- Strengths: ...
- Weaknesses: ...

### 4. Limitations & Future Work
- ...

### 5. Overall Assessment
- Score: ★★★★☆
- Recommendation: ...
```

## Notes

- Always verify paper names via `search_papers()` before reading.
- If a section is unusually large, read it in multiple calls with specific subsection names.
- When comparing to related work, use `arxiv__search_papers` to find relevant arXiv papers.
