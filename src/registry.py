import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class CardMetadata:
    card_id: str
    bank: str
    card_name: str
    file_path: str
    file_size_bytes: int

class CardRegistry:
    """Discovers credit card insurance PDFs dynamically from bank directories."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self._cards: dict[str, CardMetadata] = {}

    def _slugify(self, text: str) -> str:
        slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
        return re.sub(r"[-\s]+", "_", slug)

    def _format_display_name(self, filename: str) -> str:
        stem = Path(filename).stem
        cleaned = re.sub(r"[-_]+", " ", stem)
        cleaned = re.sub(r"\b(COI|En|Insurance|Certificate of Insurance)\b", "", cleaned, flags=re.IGNORECASE)
        return " ".join(cleaned.split())

    def scan(self) -> List[CardMetadata]:
        self._cards.clear()
        if not self.base_dir.exists():
            return []

        for bank_dir in self.base_dir.iterdir():
            if bank_dir.is_dir() and not bank_dir.name.startswith((".", "docs", "src", "tests", "web")):
                bank_name = bank_dir.name.upper()
                for pdf_file in bank_dir.glob("*.pdf"):
                    card_id = self._slugify(pdf_file.stem)
                    card_name = self._format_display_name(pdf_file.name)
                    meta = CardMetadata(
                        card_id=card_id,
                        bank=bank_name,
                        card_name=card_name,
                        file_path=str(pdf_file.resolve()),
                        file_size_bytes=pdf_file.stat().st_size
                    )
                    self._cards[card_id] = meta

        return sorted(list(self._cards.values()), key=lambda x: (x.bank, x.card_name))

    def get_card(self, card_id: str) -> Optional[CardMetadata]:
        if not self._cards:
            self.scan()
        return self._cards.get(card_id)
