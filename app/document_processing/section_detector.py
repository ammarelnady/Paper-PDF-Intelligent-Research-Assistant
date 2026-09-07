import re


class SectionDetector:
    """Detect likely section headings in academic papers."""

    COMMON_HEADINGS = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "methodology",
        "methods",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "references",
        "acknowledgements",
        "acknowledgments",
    }

    def is_heading(self, line: str) -> bool:
        line = line.strip()

        if not line:
            return False

        # Ignore page numbers
        if line.isdigit():
            return False

        normalized = line.lower().rstrip(":")

        # Ignore conference headers / footers
        if "conference on neural information processing systems" in normalized:
            return False

        # Known section names
        if normalized in self.COMMON_HEADINGS:
            return True

        # Numbered headings
        match = re.match(r"^\d+(?:\.\d+)*\.?\s+(.+)$",line)

        if not match:
            return False

        title = match.group(1).strip()

        # Heading should not be too long
        if len(title) > 80:
            return False

        # Avoid normal sentences
        if title.endswith("."):
            return False

        # Avoid scientific notation
        if re.search(r"\d+\s*[·×x]\s*10", title):
            return False

        # Avoid years such as 2014, 2017, etc.
        if re.match(r"^(19|20)\d{2}\b", line):
            return False

        # Must contain letters
        if not re.search(r"[A-Za-z]", title):
            return False

        return True

    def detect_sections(
        self,
        text: str,
        current_section: str = "Unknown",
    ) -> tuple[list[tuple[str, str]], str]:

        lines = text.splitlines()

        sections = []
        current_text = []

        for line in lines:

            if self.is_heading(line):

                if current_text:
                    section_text = "\n".join(current_text).strip()

                    if section_text:
                        sections.append(
                            (current_section, section_text)
                        )

                current_section = line.strip()
                current_text = []

            else:
                current_text.append(line)

        if current_text:
            section_text = "\n".join(current_text).strip()

            if section_text:
                sections.append(
                    (current_section, section_text)
                )

        return sections, current_section