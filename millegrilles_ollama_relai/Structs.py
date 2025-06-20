from typing import Optional

from pydantic import BaseModel


class SummaryText(BaseModel):
  summary: str
  language: Optional[str]
  tags: Optional[list[str]]
