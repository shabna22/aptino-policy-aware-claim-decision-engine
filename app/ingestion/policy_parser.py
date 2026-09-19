import pymupdf
from pathlib import Path


POLICY_PATH = Path(
    "policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
)


def extract_policy_pages():
    """Extract text from every page of the policy PDF."""

    document = pymupdf.open(POLICY_PATH)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")

        pages.append(
            {
                "page": page_number,
                "text": text.strip(),
            }
        )

    document.close()

    return pages


if __name__ == "__main__":
    pages = extract_policy_pages()

    print(f"Total pages extracted: {len(pages)}")

    for page in pages[:2]:
        print("\n" + "=" * 60)
        print(f"PAGE {page['page']}")
        print("=" * 60)
        print(page["text"][:1000])