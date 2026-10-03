from typing import Optional, TypedDict

from pydantic import BaseModel


class SummaryText(BaseModel):
    summary: str
    # language: Optional[str]
    tags: Optional[list[str]]


class SummaryKeywords(BaseModel):
    t: Optional[str]
    l: str
    url: Optional[str]


class LinkIdPicker(BaseModel):
    link_ids: list[int]


class MardownTextResponse:

    def __init__(self, text: str, complete_block=False):
        self.text = text
        self.complete_block = complete_block


class KnowledgeBaseSearchResponse:

    def __init__(self, search_url: Optional[str], reference_title: str, reference_url: str, summary: str):
        self.search_url = search_url
        self.reference_title = reference_title
        self.reference_url = reference_url
        self.summary = summary


class MatchResult(BaseModel):
    summary: str
    match: bool


class OllamaRelaiConfigurationPropertiesValue(TypedDict):
    text: Optional[str]
    inumber: Optional[int]
    fnumber: Optional[float]


class OllamaRelaiConfigurationProperties(TypedDict):
    file_id: str
    key: str
    value: OllamaRelaiConfigurationPropertiesValue
    last_modified: int


class OllamaRelaiConfigurationFile(TypedDict):
    file_id: str
    filename: str
    key_id: str
    last_modified: int
    list: list[OllamaRelaiConfigurationProperties]



def get_property_text(properties: dict, key: str) -> Optional[str]:
    try:
        return properties[key]['text']
    except KeyError:
        return None

def get_property_int(properties: dict, key: str) -> Optional[int]:
    try:
        return properties[key]['inumber']
    except KeyError:
        return None

def get_property_float(properties: dict, key: str) -> Optional[float]:
    try:
        return properties[key]['fnumber']
    except KeyError:
        return None
