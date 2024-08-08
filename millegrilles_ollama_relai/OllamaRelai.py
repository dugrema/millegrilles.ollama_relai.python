import asyncio
import logging
import signal

from typing import Optional

from millegrilles_messages.MilleGrillesConnecteur import MilleGrillesConnecteur
from millegrilles_ollama_relai.Context import OllamaRelaiContext
from millegrilles_ollama_relai.CommandHandler import CommandHandler
from millegrilles_ollama_relai.QueryHandler import QueryHandler


class OllamaRelai:

    def __init__(self, context: OllamaRelaiContext):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context

        # Must init after from asyncio loop is started
        self.__stop_event: Optional[asyncio.Event] = None
        self.__rabbitmq_dao: Optional[MilleGrillesConnecteur] = None
        self.__command_handler: Optional[CommandHandler] = None
        self.__query_handler: Optional[QueryHandler] = None
        self.__loop = None

    @property
    def context(self):
        return self.__context

    async def __setup(self):
        self.__loop = asyncio.get_event_loop()
        self.__stop_event = asyncio.Event()
        await self.__context.reload_configuration()
        self.__query_handler = QueryHandler(self.__context)
        self.__command_handler = CommandHandler(self, self.__query_handler)
        self.__rabbitmq_dao = MilleGrillesConnecteur(self.__stop_event, self.__context, self.__command_handler)

    async def run(self):
        self.__logger.info("Setup")
        await self.__setup()

        self.__logger.info("Start running")
        tasks = [
            asyncio.create_task(self.__context.run(self.__stop_event, self.__rabbitmq_dao), name='context'),
            asyncio.create_task(self.__rabbitmq_dao.run(), name="mq"),
            asyncio.create_task(self.__query_handler.run(self.__stop_event), name="query_handler"),
        ]

        await asyncio.gather(*tasks)

        self.__logger.info("End running")

    def exit_gracefully(self, signum=None, frame=None):
        self.__logger.info("Close application, signal: %d" % signum)
        self.__loop.call_soon_threadsafe(self.__stop_event.set)


async def run(context: OllamaRelaiContext):
    relai = OllamaRelai(context)
    signal.signal(signal.SIGINT, relai.exit_gracefully)
    signal.signal(signal.SIGTERM, relai.exit_gracefully)
    await relai.run()
