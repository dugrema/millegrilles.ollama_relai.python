import asyncio
import logging
import pathlib

from asyncio import TaskGroup
from typing import Callable, Awaitable, Optional

from millegrilles_messages.bus.BusContext import ForceTerminateExecution
from millegrilles_messages.messages import Constantes
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_messages.structs.Filehost import Filehost
from millegrilles_messages.Filehost import FilehostConnection
from millegrilles_ollama_relai.DocumentIndexHandler import DocumentIndexHandler
from millegrilles_ollama_relai.OllamaChatHandler import OllamaChatHandler
from millegrilles_ollama_relai.OllamaContext import OllamaContext, RagConfiguration, UrlConfiguration
from millegrilles_ollama_relai.OllamaInstanceManager import OllamaInstanceManager, OllamaInstance
from millegrilles_ollama_relai.OllamaTools import OllamaToolHandler


class OllamaManager:

    def __init__(self, context: OllamaContext, ollama_instances: OllamaInstanceManager,
                 attachment_handler: FilehostConnection, tool_handler: OllamaToolHandler, chat_handler: OllamaChatHandler,
                 document_handler: DocumentIndexHandler):
        self.__logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
        self.__context = context
        self.__ollama_instances = ollama_instances
        self.__attachment_handler = attachment_handler
        self.__tool_handler = tool_handler
        self.__chat_handler = chat_handler
        self.__document_handler = document_handler

        self.__filehost_listeners: list[Callable[[Optional[Filehost]], Awaitable[None]]] = list()

        self.__load_ai_configuration_event = asyncio.Event()
        self.__load_filehost_event = asyncio.Event()
        self.__ollama_available = False

    @property
    def context(self):
        return self.__context

    async def setup(self):
        # Create staging folders
        #self.__context.dir_ollama_staging.mkdir(parents=True, exist_ok=True)
        configuration = self.__context.configuration
        dir_rag = pathlib.Path(configuration.dir_rag)
        dir_rag.mkdir(parents=True, exist_ok=True)

    async def __stop_thread(self):
        await self.__context.wait()
        # Free threads
        self.__load_ai_configuration_event.set()
        self.__load_filehost_event.set()
        await asyncio.sleep(0.5)

    async def run(self):
        self.__logger.debug("OllamaManager Starting")
        try:
            async with TaskGroup() as group:
                group.create_task(self.__reload_filehost_thread())
                group.create_task(self.__reload_ai_configuration_thread())
                group.create_task(self.__ollama_watchdog_thread())
                group.create_task(self.__stop_thread())
        except* (asyncio.CancelledError, ForceTerminateExecution):
            if self.__context.stopping is False:
                self.__logger.warning("Ollama manager cancelled / forced to terminate without setting context to stop")
                self.__context.stop()
            else:
                self.__logger.info("OllamaManager tasks cancelled, stopping")
        self.__logger.debug("OllamaManager Done")

    def add_filehost_listener(self, listener: Callable[[Optional[Filehost]], Awaitable[None]]):
        self.__filehost_listeners.append(listener)

    async def __reload_filehost_thread(self):
        while self.__context.stopping is False:
            self.__load_filehost_event.clear()
            try:
                await self.reload_filehost_configuration()
            except asyncio.TimeoutError as e:
                self.__logger.error("Error loading filehost configuration: %s" % e)
                await self.__context.wait(15)
                continue  # Loop immediately

            try:
                await asyncio.wait_for(self.__load_filehost_event.wait(), 900)
            except asyncio.TimeoutError:
                pass  # Loop

        self.__logger.info("__reload_filehost_thread Stopping")

    async def reload_filehost_configuration(self):
        await self.__context.reload_filehost_configuration()

        for l in self.__filehost_listeners:
            await l(self.__context.filehost)

    async def trigger_reload_ai_configuration(self):
        self.__logger.info("Reloading AI Configuration on event")
        self.__load_ai_configuration_event.set()

    async def __reload_ai_configuration_thread(self):
        while self.__context.stopping is False:
            try:
                self.__load_ai_configuration_event.clear()
                await self.__reload_ai_configuration()
            except asyncio.TimeoutError:
                self.__logger.exception("Error loading ai configuration")
                await self.__context.wait(20)

            try:
                await asyncio.wait_for(self.__load_ai_configuration_event.wait(), 300)
            except asyncio.TimeoutError:
                pass  # Loop

        self.__logger.info("__reload_ai_configuration_thread Stopping")

    async def register_chat(self, message: MessageWrapper):
        return await self.__chat_handler.register_query(message)

    async def process_chat(self, instance: OllamaInstance, message: MessageWrapper):
        return await self.__chat_handler.process_chat(instance, message)

    async def cancel_chat(self, message: MessageWrapper):
        try:
            # Block processing from redis
            chat_id = message.parsed['chat_id']
            await self.__ollama_instances.claim_query(chat_id)
        except Exception:
            pass  # Already locked (processing)

        # Cancel the chat when already processing - this also puts a lock in memory in case redis is not available
        return await self.__chat_handler.cancel_chat(message)

    async def register_rag_query(self, message):
        return await self.__document_handler.register_rag(message)

    async def query_rag(self, instance: OllamaInstance, message: MessageWrapper):
        return await self.__document_handler.query_rag(instance, message)

    async def trigger_rag_indexing(self, delay: Optional[float] = None):
        await self.__document_handler.trigger_indexing(delay=delay)

    async def __reload_ai_configuration(self):
        producer = await self.context.get_producer()
        response = await producer.request(dict(), "AiLanguage", "getConfiguration", exchange=Constantes.SECURITE_PRIVE)
        parsed = response.parsed

        try:
            chat_configuration = parsed['default']
        except (TypeError, KeyError):
            self.__context.chat_configuration = None  # No information
        else:
            self.__context.chat_configuration = chat_configuration

        try:
            model_configuration = parsed['models']
        except (TypeError, KeyError):
            self.__context.model_configuration = None  # No information
        else:
            self.__context.model_configuration = model_configuration

        try:
            urls = parsed['ollama_urls']['urls']
        except (TypeError, KeyError):
            pass  # No URL information
        else:
            await self.__ollama_instances.update_instance_list(urls)

        try:
            rag_configuration: RagConfiguration = parsed['rag']
        except (TypeError, KeyError):
            self.__context.rag_configuration = None  # No information
        else:
            self.__context.rag_configuration = rag_configuration

        try:
            url_configuration: UrlConfiguration = parsed['urls']
        except (TypeError, KeyError):
            self.__context.url_configuration = None  # No information
        else:
            self.__context.url_configuration = url_configuration

        # For initial configuration load
        self.__context.ai_configuration_loaded.set()

    async def __ollama_watchdog_thread(self):
        """ Regularly checks status of ollama connection. """
        while self.__context.stopping is False:
            try:
                await asyncio.wait_for(self.__context.ai_configuration_loaded.wait(), 2)
            except asyncio.TimeoutError:
                continue  # Retry

            try:
                await self.__context.wait(10)
            except ForceTerminateExecution:
                pass

        self.__logger.info("__ollama_watchdog_thread Stopping")
