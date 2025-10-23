from typing import Dict
from edgar import set_identity, Company


class EdgarClient:
    # Esses itens contêm as informações mais relevantes para análise financeira
    FORM_ITEMS = {"10-K": ["1", "1A", "7", "8", "9A"], "10-Q": ["1", "2", "3", "4"]}

    def __init__(self, email: str):
        set_identity(email)

    def fetch_filing_data(self, ticker: str, form_type: str) -> Dict[str, any]:
        company = Company(ticker)
        filing = company.get_filings(form=form_type).latest()

        metadata = {
            "ticker": ticker,
            "company_name": filing.company,
            "report_date": str(filing.report_date),
            "form_type": filing.form,
        }

        filing_obj = filing.obj()
        items = {}

        for item_num in self.FORM_ITEMS[form_type]:
            item_key = f"Item {item_num}"
            try:
                items[item_key] = filing_obj[item_key]
            except (KeyError, IndexError):
                continue

        return {"metadata": metadata, "items": items}

    def get_combined_text(self, data: Dict) -> str:
        texts = []
        for item_name, item_content in data["items"].items():
            texts.append(f"## {item_name}\n\n{item_content}")

        return "\n\n".join(texts)
