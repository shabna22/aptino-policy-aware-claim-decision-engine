from pathlib import Path
import sys
import re

sys.path.append(str(Path(__file__).parent))

from policy_parser import extract_policy_pages


def create_chunks(pages, max_chars=1500):
    """
    Create meaningful policy chunks while preserving
    page number, section heading, and chunk ID.
    """

    chunks = []

    for page in pages:
        page_number = page["page"]
        text = page["text"]

        # Clean excessive whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        # Split into paragraphs
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        current_text = ""
        section = f"Page {page_number}"
        chunk_number = 1

        for paragraph in paragraphs:

            # Detect likely section headings
            if (
                len(paragraph) < 120
                and (
                    paragraph.isupper()
                    or paragraph.endswith(":")
                    or "SECTION" in paragraph.upper()
                )
            ):
                section = paragraph

            # Add paragraph to current chunk
            if current_text:
                candidate = current_text + "\n\n" + paragraph
            else:
                candidate = paragraph

            # Create chunk when it becomes reasonably large
            if len(candidate) > max_chars and current_text:

                chunks.append(
                    {
                        "chunk_id": f"policy_p{page_number}_c{chunk_number}",
                        "page": page_number,
                        "section": section,
                        "text": current_text,
                    }
                )

                chunk_number += 1
                current_text = paragraph

            else:
                current_text = candidate

        # Save remaining text from the page
        if current_text:

            chunks.append(
                {
                    "chunk_id": f"policy_p{page_number}_c{chunk_number}",
                    "page": page_number,
                    "section": section,
                    "text": current_text,
                }
            )

    return chunks


if __name__ == "__main__":

    import json

    pages = extract_policy_pages()

    chunks = create_chunks(pages)

    output_path = Path("app/ingestion/policy_chunks.json")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunks saved to: {output_path}")