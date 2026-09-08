from pathlib import Path
import re

import pymupdf

from app.models.document import PageContent


class PDFExtractor:
    """Extract and conservatively normalize text from PDF pages."""

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Conservatively normalize extracted PDF text.

        Preserves paragraph boundaries, punctuation, numbers,
        URLs, scientific notation, and equation-like text.
        """

        # Normalize line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize horizontal whitespace without removing newlines.
        text = re.sub(r"[ \t\f\v]+", " ", text)

        # Clean whitespace around each line.
        lines = []

        for line in text.split("\n"):
            line = line.strip()

            if line:
                lines.append(line)
            elif lines and lines[-1] != "":
                # Keep paragraph boundaries.
                lines.append("")

        # Remove leading/trailing blank lines.
        while lines and lines[0] == "":
            lines.pop(0)

        while lines and lines[-1] == "":
            lines.pop()

        # Avoid excessive blank lines.
        normalized = "\n".join(lines)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        return normalized

    def extract(
        self,
        pdf_path: str | Path
    ) -> list[PageContent]:

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        pages = []

        with pymupdf.open(pdf_path) as pdf:

            for page_index, page in enumerate(pdf):

                raw_text = page.get_text("text")
                text = self._normalize_text(raw_text)

                pages.append(
                    PageContent(
                        page_number=page_index + 1,
                        text=text,
                    )
                )

        return pages