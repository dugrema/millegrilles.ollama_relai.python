import logging
import pathlib

from asyncio import TaskGroup
from typing import Callable, Awaitable, Optional

from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_messages.structs.Filehost import Filehost
from millegrilles_ollama_relai.AttachmentHandler import AttachmentHandler
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.QueryHandler import QueryHandler


class OllamaManager:

    def __init__(self, context: OllamaContext, query_handler: QueryHandler, attachment_handler: AttachmentHandler):
        self.__logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
        self.__context = context
        self.__query_handler = query_handler
        self.__attachment_handler = attachment_handler

        self.__filehost_listeners: list[Callable[[Optional[Filehost]], Awaitable[None]]] = list()

    @property
    def context(self):
        return self.__context

    async def setup(self):
        # Create staging folders
        #self.__context.dir_ollama_staging.mkdir(parents=True, exist_ok=True)
        configuration = self.__context.configuration
        dir_rag = pathlib.Path(configuration.dir_rag)
        dir_rag.mkdir(parents=True, exist_ok=True)
        pass

    async def run(self):
        self.__logger.debug("OllamaManager Starting")
        async with TaskGroup() as group:
            group.create_task(self.__reload_filehost_thread())
            group.create_task(self.__staging_cleanup())
        self.__logger.debug("OllamaManager Done")

    def add_filehost_listener(self, listener: Callable[[Optional[Filehost]], Awaitable[None]]):
        self.__filehost_listeners.append(listener)

    async def __reload_filehost_thread(self):
        while self.__context.stopping is False:
            try:
                await self.reload_filehost_configuration()
                await self.__context.wait(900)
            except:
                self.__logger.exception("Error loading filehost configuration")
                await self.__context.wait(30)

    async def reload_filehost_configuration(self):
        producer = await self.__context.get_producer()
        response = await producer.request(
            dict(), 'CoreTopologie', 'getFilehostForInstance', exchange="1.public")

        try:
            filehost_response = response.parsed
            filehost_dict = filehost_response['filehost']
            filehost = Filehost.load_from_dict(filehost_dict)
            self.__context.filehost = filehost
        except:
            self.__logger.exception("Error loading filehost")
            self.__context.filehost = None

        for l in self.__filehost_listeners:
            await l(self.__context.filehost)

    async def __staging_cleanup(self):
        while self.__context.stopping is False:
            # TODO - cleanup

            await self.__context.wait(300)

    async def handle_volalile_request(self, message: MessageWrapper):
        return await self.__query_handler.handle_requests(message)

    async def handle_volalile_query(self, message: MessageWrapper):
        return await self.__query_handler.handle_query(message)

    async def process_query(self, message: MessageWrapper):
        return await self.__query_handler.process_query(message)
