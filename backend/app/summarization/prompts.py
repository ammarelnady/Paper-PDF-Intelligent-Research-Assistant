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
2. If a piece of information (e.g. limitations or problem statement) is not mentioned in the text, state "Not explicitly mentioned in the provided text."
3. Maintain objective, academic tone.
4. Reference section names or key terms from the text directly.
"""

# Structured Overall Summary Prompt
OVERALL_PAPER_SUMMARY_PROMPT = """Analyze the following scientific paper content and generate a comprehensive, structured summary.

=== PAPER CONTENT ===
{context}

=== INSTRUCTIONS ===
Provide your response strictly in the following JSON format:
{{
    "summary": "A concise 2-3 paragraph executive summary of the entire paper, including background, core idea, and outcome.",
    "problem_statement": "The core problem, research question, or challenge the authors are addressing.",
    "methodology": "The proposed approach, model architecture, dataset, or experimental technique.",
    "key_contributions": [
        "First key contribution or novelty",
        "Second key contribution",
        "Third key contribution"
    ],
    "findings": "Key experimental results, performance metrics, discoveries, or empirical outcomes.",
    "limitations": "Limitations, constraints, assumptions, or future work mentioned by the authors."
}}
"""

# Section-Level Summary Prompt
SECTION_SUMMARY_PROMPT = """You are summarizing the "{section_name}" section of a scientific research paper.

=== SECTION CONTENT ===
{context}

=== INSTRUCTIONS ===
Provide a concise, faithful summary (1-2 paragraphs) capturing the key points of this section.
Ensure the summary is strictly grounded in the provided section text without external speculation.
"""

# Extractive / Fallback Templates
KEY_CONTRIBUTIONS_PROMPT = """List the 3-5 main contributions and novel ideas introduced in this paper based on the text below:
{context}
"""

PROBLEM_STATEMENT_PROMPT = """State clearly what specific problem or research gap this paper addresses:
{context}
"""

METHODOLOGY_PROMPT = """Summarize the proposed methodology, architecture, and theoretical formulation:
{context}
"""

FINDINGS_PROMPT = """Summarize the main experimental results, findings, and evaluation metrics:
{context}
"""

LIMITATIONS_PROMPT = """What limitations, trade-offs, or threats to validity are identified in this paper?
{context}
"""
