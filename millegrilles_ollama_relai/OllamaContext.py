import asyncio

import aiohttp
import logging
import ssl
import pathlib

from typing import Optional, TypedDict
from urllib.parse import urlparse

from millegrilles_messages.bus.BusContext import MilleGrillesBusContext
from millegrilles_messages.bus.PikaConnector import MilleGrillesPikaConnector
from millegrilles_messages.structs.Filehost import Filehost
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
        self.__bus_connector: Optional[MilleGrillesPikaConnector] = None
        self.__filehost: Optional[Filehost] = None
        self.__filehost_url: Optional[str] = None
        self.__tls_method: Optional[str] = None
        self.__ssl_context_filehost: Optional[ssl.SSLContext] = None

        self.ai_configuration_loaded = asyncio.Event()
        self.rag_configuration: Optional[RagConfiguration] = None
        self.chat_configuration: Optional[ChatConfiguration] = None

    @property
    def configuration(self) -> OllamaConfiguration:
        return super().configuration

    @property
    def bus_connector(self):
        return self.__bus_connector

    @bus_connector.setter
    def bus_connector(self, value: MilleGrillesPikaConnector):
        self.__bus_connector = value

    async def get_producer(self):
        return await self.__bus_connector.get_producer()

    @property
    def filehost(self) -> Optional[Filehost]:
        return self.__filehost

    @filehost.setter
    def filehost(self, value: Filehost):
        self.__filehost = value

        # Pick URL
        url, tls_method = OllamaContext.__load_url(value)
        self.__filehost_url = url.geturl()
        self.__tls_method = tls_method

    @property
    def filehost_url(self):
        return self.__filehost_url

    @property
    def tls_method(self):
        return self.__tls_method

    def get_tcp_connector(self) -> aiohttp.TCPConnector:
        # Prepare connection information (SSL)
        ssl_context = None
        verify = True
        if self.__tls_method == 'millegrille':
            ssl_context = self.ssl_context
        elif self.__tls_method == 'nocheck':
            verify = False

        connector = aiohttp.TCPConnector(ssl=ssl_context, verify_ssl=verify)

        return connector

    @property
    def dir_ollama_staging(self):
        return pathlib.Path(self.configuration.dir_staging, 'ollama')

    @staticmethod
    def __load_url(filehost: Filehost):
        if filehost.url_external:
            url = urlparse(filehost.url_external)
            tls_method = filehost.tls_external
        elif filehost.url_internal:
            url = urlparse(filehost.url_internal)
            tls_method = 'millegrille'
        else:
            raise ValueError("No valid URL")
        return url, tls_method
