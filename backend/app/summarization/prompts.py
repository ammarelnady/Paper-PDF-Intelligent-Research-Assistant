"""
Summarization Prompts and Templates for Scientific Papers.

Includes structured prompts for overall paper summarization, section-level summaries,
key contributions, methodology, findings, limitations, and anti-hallucination grounding.
"""

# Groundedness & System Instructions
GROUNDEDNESS_SYSTEM_INSTRUCTION = """You are an expert scientific paper analysis assistant.
Your job is to produce clear, accurate, and faithful summaries of research papers based STRICTLY on the provided text.

CRITICAL RULES:
1. Do NOT make up facts, numbers, citations, or conclusions not supported by the context.
2. Ignore author lists, emails, affiliations, arXiv watermarks, copyright notices, and table fragments.
3. If limitations are not explicitly stated in a dedicated section, infer the trade-offs, constraints, or computational complexity mentioned in the paper, or explicitly state "Not explicitly stated as a standalone section in the paper." (Never return null or None).
4. Clearly state the actual research problem (what obstacle or limitation in prior work the authors are solving).
5. Maintain an objective, academic tone.
"""

# Structured Overall Summary Prompt
OVERALL_PAPER_SUMMARY_PROMPT = """Analyze the following scientific paper content and generate a comprehensive, structured summary.

=== PAPER CONTENT ===
{context}

=== INSTRUCTIONS ===
Provide your response strictly in the following JSON format:
{{
    "summary": "A cohesive, well-written 2-3 paragraph executive summary explaining the background, core proposed methodology, and key empirical outcomes without copying raw author lists or table headers.",
    "problem_statement": "The specific research problem, computational bottleneck, or challenge the authors are addressing.",
    "methodology": "The proposed approach, architecture, algorithm, or experimental setup.",
    "key_contributions": [
        "First primary contribution or novel architecture",
        "Second key contribution",
        "Third key contribution or benchmark result"
    ],
    "findings": "Key experimental results, benchmarks (e.g. BLEU/accuracy), and findings.",
    "limitations": "Limitations, trade-offs (e.g. memory/computation complexity), or future work discussed by the authors."
}}
"""

# Section-Level Summary Prompt
SECTION_SUMMARY_PROMPT = """You are summarizing the "{section_name}" section of a scientific research paper.

=== SECTION CONTENT ===
{context}

=== INSTRUCTIONS ===
Provide a concise, faithful summary (1-2 paragraphs) capturing the essential substance of this section.
Ensure the summary is strictly grounded in the provided section text without external speculation or noise.
"""

# Extractive / Targeted Prompts
KEY_CONTRIBUTIONS_PROMPT = """List the 3-5 main contributions and novel ideas introduced in this paper based on the text below:
{context}
"""

PROBLEM_STATEMENT_PROMPT = """State clearly what specific problem, bottleneck, or research gap this paper addresses:
{context}
"""

METHODOLOGY_PROMPT = """Summarize the proposed methodology, architecture, and theoretical formulation:
{context}
"""

FINDINGS_PROMPT = """Summarize the main experimental results, findings, and evaluation metrics:
{context}
"""

LIMITATIONS_PROMPT = """What limitations, trade-offs, computational complexities, or threats to validity are discussed in this paper?
{context}
"""
