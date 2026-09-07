from pathlib import Path

import pymupdf

from app.models.document import PageContent


class PDFExtractor:
    """Extract text from PDF pages."""

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

                text = page.get_text("text").strip()

                pages.append(
                    PageContent(
                        page_number=page_index + 1,
                        text=text,
                    )
                )

        return pages