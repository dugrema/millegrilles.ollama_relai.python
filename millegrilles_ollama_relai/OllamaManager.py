import asyncio
import logging
import pathlib

from asyncio import TaskGroup
from typing import Callable, Awaitable, Optional

from millegrilles_messages.bus.BusContext import ForceTerminateExecution
from millegrilles_messages.messages import Constantes
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_messages.structs.Filehost import Filehost
from millegrilles_ollama_relai.AttachmentHandler import AttachmentHandler
from millegrilles_ollama_relai.DocumentIndexHandler import DocumentIndexHandler
from millegrilles_ollama_relai.OllamaChatHandler import OllamaChatHandler
from millegrilles_ollama_relai.OllamaContext import OllamaContext, RagConfiguration
from millegrilles_ollama_relai.OllamaInstanceManager import OllamaInstanceManager, OllamaInstance
from millegrilles_ollama_relai.OllamaTools import OllamaToolHandler


class OllamaManager:

    def __init__(self, context: OllamaContext, ollama_instances: OllamaInstanceManager,
                 attachment_handler: AttachmentHandler, tool_handler: OllamaToolHandler, chat_handler: OllamaChatHandler,
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
        producer = await self.__context.get_producer()
        response = await producer.request(
            dict(), 'CoreTopologie', 'getFilehostForInstance', exchange="1.public")

        try:
            filehost_response = response.parsed
            filehost_dict = filehost_response['filehost']
            filehost = Filehost.load_from_dict(filehost_dict)
            self.__context.filehost = filehost
        except (KeyError, AttributeError, ValueError):
            self.__logger.exception("Error loading filehost")
            self.__context.filehost = None

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

    # async def handle_volalile_request(self, message: MessageWrapper):
    #     return await self.__message_handler.handle_requests(message)

    # async def handle_volalile_commands(self, message: MessageWrapper):
    #     return await self.__message_handler.handle_commands(message)

    async def register_chat(self, message: MessageWrapper):
        return await self.__chat_handler.register_query(message)

    async def process_chat(self, instance: OllamaInstance, message: MessageWrapper):
        return await self.__chat_handler.process_chat(instance, message)

    async def cancel_chat(self, message: MessageWrapper):
        return await self.__chat_handler.cancel_chat(message)

    async def register_rag_query(self, message):
        return await self.__document_handler.register_rag(message)

    async def query_rag(self, instance: OllamaInstance, message: MessageWrapper):
        return await self.__document_handler.query_rag(instance, message)

    async def trigger_rag_indexing(self):
        await self.__document_handler.trigger_indexing()

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
            urls = parsed['ollama_urls']['urls']
        except (TypeError, KeyError):
            pass  # No URL information
        else:
            self.__ollama_instances.update_instance_list(urls)

        try:
            rag_configuration: RagConfiguration = parsed['rag']
        except (TypeError, KeyError):
            self.__context.rag_configuration = None  # No information
        else:
            self.__context.rag_configuration = rag_configuration

        # For initial configuration load
        self.__context.ai_configuration_loaded.set()

    async def __ollama_watchdog_thread(self):
        """ Regularly checks status of ollama connection. """
        while self.__context.stopping is False:
            try:
                await asyncio.wait_for(self.__context.ai_configuration_loaded.wait(), 2)
            except asyncio.TimeoutError:
                continue  # Retry

            # # available = self.__ollama_instances.ollama_status is not False
            # # await self.__check_ollama_status()
            # now_available = self.__ollama_instances.ollama_ready is not False
            # if self.__ollama_available != now_available:
            #     self.__ollama_available = now_available  # Toggle local flag for next check
            #     self.__logger.info("ollama Status now %s" % now_available)
            #     producer = await self.__context.get_producer()
            #     status_event = {'event_type': 'availability', 'available': now_available}
            #     await producer.event(
            #         status_event, 'ollama_relai', 'status',
            #         exchange=Constantes.SECURITE_PRIVE)

            try:
                await self.__context.wait(10)
            except ForceTerminateExecution:
                pass

        self.__logger.info("__ollama_watchdog_thread Stopping")

    # async def __check_ollama_status(self):
    #     instances = self.__ollama_instances.ollama_instances
    #
    #     models = set()
    #     for instance in instances:
    #         self.__logger.debug(f"Checking with {instance.url}")
    #         client = instance.get_async_client(self.context.configuration)
    #         status = False
    #         try:
    #             # Test connection by getting currently loaded model information
    #             async with instance.semaphore:
    #                 instance.ollama_status = await client.ps()
    #                 instance.ollama_models = await client.list()
    #                 for model in instance.ollama_models.models:
    #                     try:
    #                         self.__ollama_instances.ollama_model_params[model.model]
    #                     except KeyError:
    #                         params = dict()
    #                         model_info = await client.show(model.model)
    #                         params['capabilities'] = model_info.capabilities
    #                         self.__ollama_instances.ollama_model_params[model.model] = params
    #
    #                 instance_models = [m.model for m in instance.ollama_models.models]
    #                 models.update(instance_models)
    #                 status = True
    #                 self.__logger.debug(f"Connection OK: {instance.url}")
    #         except (httpx.ConnectError, ConnectionError) as e:
    #             # Failed to connect
    #             if self.__logger.isEnabledFor(logging.DEBUG):
    #                 self.__logger.exception(f"Connection error on {instance.url}")
    #             else:
    #                self.__logger.info(f"Connection error on {instance.url}: %s" % str(e))
    #             instance.status = None  # Reset status, avoids picking this instance up
    #
    #         # Indicates at least one instance is responsive
    #         self.__context.ollama_status = status
    #         ollama_models = list()
    #         for m in models:
    #             model_info = {'name': m}
    #             try:
    #                 params = self.__ollama_instances.ollama_model_params[m]
    #                 model_info.update(params)
    #             except KeyError:
    #                 pass
    #             ollama_models.append(model_info)
    #
    #         # self.__context.ollama_models = [{'name': m} for m in models]
    #         self.__context.ollama_models = ollama_models
