from typing import Optional

from pydantic import BaseModel


class SummaryText(BaseModel):
  summary: str
  # language: Optional[str]
  tags: Optional[list[str]]


class SummaryKeywords(BaseModel):
  s: str
  t: Optional[str]
  l: str


class LinkIdPicker(BaseModel):
  link_id: int
