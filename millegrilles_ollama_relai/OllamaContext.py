import asyncio

import logging
import pathlib

from typing import Optional, TypedDict

from millegrilles_messages.bus.BusContext import MilleGrillesBusContext
from millegrilles_messages.bus.PikaConnector import MilleGrillesPikaConnector
from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration

LOGGER = logging.getLogger(__name__)

class ChatConfiguration(TypedDict):
    default_model: Optional[str]
    chat_context_length: Optional[int]

class RagConfiguration(TypedDict):
    model_embedding_name: Optional[str]
    model_query_name: Optional[str]
    context_len: Optional[int]
    document_chunk_len: Optional[int]
    document_overlap_len: Optional[int]


class OllamaContext(MilleGrillesBusContext):

    def __init__(self, configuration: OllamaConfiguration):
        super().__init__(configuration)
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        # self.__bus_connector: Optional[MilleGrillesPikaConnector] = None

        self.ai_configuration_loaded = asyncio.Event()
        self.rag_configuration: Optional[RagConfiguration] = None
        self.chat_configuration: Optional[ChatConfiguration] = None

    @property
    def configuration(self) -> OllamaConfiguration:
        return super().configuration

    @property
    def dir_ollama_staging(self):
        return pathlib.Path(self.configuration.dir_staging, 'ollama')
