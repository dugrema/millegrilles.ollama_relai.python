import asyncio

import aiohttp
import logging
import ssl
import pathlib

from ollama import AsyncClient, ProcessResponse, ListResponse
from typing import Optional, Any, Union, Mapping, TypedDict
from urllib.parse import urlparse

from millegrilles_messages.bus.BusContext import MilleGrillesBusContext
from millegrilles_messages.bus.PikaConnector import MilleGrillesPikaConnector
from millegrilles_messages.structs.Filehost import Filehost
from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration

LOGGER = logging.getLogger(__name__)

class OllamaInstance:

    def __init__(self, url: str):
        self.url = url
        self.ollama_status: Optional[ProcessResponse] = None
        self.ollama_models: Optional[ListResponse] = None
        self.semaphore = asyncio.BoundedSemaphore(1)

    def is_available(self) -> bool:
        return self.semaphore.locked()

    def get_client_options(self, configuration: OllamaConfiguration) -> dict:
        connection_url = self.url
        if connection_url.lower().startswith('https://'):
            # Use a millegrille certificate authentication
            cert = (configuration.cert_path, configuration.key_path)
            params = {'host':connection_url, 'verify':configuration.ca_path, 'cert':cert}
        else:
            params = {'host':connection_url}
        return params

    def get_async_client(self, configuration: OllamaConfiguration) -> AsyncClient:
        options = self.get_client_options(configuration)
        return AsyncClient(**options)


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
        # Semaphore to only allow 1 http request at a time to the Ollama backend

        self.__instances: list[OllamaInstance] = list()
        self.ollama_status: bool = False
        self.ollama_models: Optional[list[Any]] = None
        self.ollama_model_params: dict[str, dict] = dict()

        # self.__ollama_http_semaphore = asyncio.BoundedSemaphore(1)
        self.ai_configuration_loaded = asyncio.Event()
        self.rag_configuration: Optional[RagConfiguration] = None


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

    # @property
    # def ollama_http_semaphore(self) -> asyncio.BoundedSemaphore:
    #     return self.__ollama_http_semaphore

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

    @property
    def ollama_instances(self) -> list[OllamaInstance]:
        return self.__instances

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

    # def get_client_options(self) -> dict:
    #     configuration = self.configuration
    #     connection_url = self.configuration.ollama_url
    #     if connection_url.lower().startswith('https://'):
    #         # Use a millegrille certificate authentication
    #         cert = (configuration.cert_path, configuration.key_path)
    #         params = {'host':connection_url, 'verify':configuration.ca_path, 'cert':cert}
    #     else:
    #         params = {'host':connection_url}
    #     return params

    # def get_async_client(self, instance: Optional[OllamaInstance] = None) -> AsyncClient:
    #     options = self.get_client_options()
    #     return AsyncClient(**options)

    def pick_ollama_instance(self, model: Optional[str] = None) -> OllamaInstance:
        if model:
            matching_instances = list()
            for instance in self.__instances:
                # Make sure the instance is available
                if instance.ollama_status is not None:
                    # Extract a list of models to check if any match the currently requested model
                    models = [m.model for m in instance.ollama_models.models]
                    if model in models:
                        matching_instances.append(instance)
        else:
            matching_instances = [i for i in self.__instances if i.ollama_status is not None]

        for instance in matching_instances:
            if instance.is_available():
                return instance

        # No instance currently free, just return the first matching instance. Implement round-robin later.
        return matching_instances[0]

    def update_instance_list(self, urls: list[str]):
        url_set = set(urls)

        to_remove = set()
        for instance in self.__instances:
            try:
                url_set.remove(instance.url)
                self.__logger.debug(f"URL {instance.url} kept")
            except KeyError:
                # This instance has been removed
                to_remove.add(instance.url)
                self.__logger.debug(f"URL {instance.url} removed")

        # Remove instances that are no longer required
        updated_list = [i for i in self.__instances if i.url not in to_remove]

        # Add missing instances
        for url in url_set:
            updated_list.append(OllamaInstance(url))
            self.__logger.debug(f"URL {url} added")

        self.__instances = updated_list
