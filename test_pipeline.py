from collections import Counter

from app.document_processing.processor import DocumentProcessor


PDF_PATH = (
    r"C:\Users\Salma\OneDrive\Desktop"
    r"\paper-research-assistant"
    r"\NIPS-2017-attention-is-all-you-need-Paper.pdf"
)


processor = DocumentProcessor()

document = processor.process(PDF_PATH)


print("=" * 60)
print("DOCUMENT INFORMATION")
print("=" * 60)

print("Document ID:", document.document_id)
print("Filename:", document.filename)
print("Total pages:", document.total_pages)
print("Total chunks:", len(document.chunks))


print("\n" + "=" * 60)
print("FIRST 10 CHUNKS")
print("=" * 60)


for chunk in document.chunks[:10]:

    print("\n--- CHUNK ---")

    print("Chunk ID:", chunk.chunk_id)
    print("Page:", chunk.page_number)
    print("Section:", chunk.section)
    print("Text:", chunk.text[:200])


print("\n" + "=" * 60)
print("SECTION DISTRIBUTION")
print("=" * 60)


section_counts = Counter(
    chunk.section
    for chunk in document.chunks
)


for section, count in section_counts.items():

    print(
        f"{section}: {count} chunks"
    )


print("\n" + "=" * 60)
print("PAGE DISTRIBUTION")
print("=" * 60)


page_counts = Counter(
    chunk.page_number
    for chunk in document.chunks
)


for page, count in sorted(page_counts.items()):

    print(
        f"Page {page}: {count} chunks"
    )