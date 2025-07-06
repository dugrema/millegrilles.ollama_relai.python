from typing import Optional, TypedDict

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


class KnowledgeBaseMarkdownText:

    def __init__(self, text: str, complete_block=False):
        self.text = text
        self.complete_block = complete_block


class KnowledgeBaseSearchResponse:

    def __init__(self, search_url: str, reference_title: str, reference_url: str):
        self.search_url = search_url
        self.reference_title = reference_title
        self.reference_url = reference_url
