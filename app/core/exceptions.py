class DocumentProcessingError(Exception):
    """Raised when a document cannot be safely processed."""


class InvalidPdfError(DocumentProcessingError):
    """Raised when an uploaded file is not a readable PDF."""
