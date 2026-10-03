import logging
from millegrilles_ollama_relai.OllamaContext import OllamaContext


class Processor:

    def __init__(self, context: OllamaContext, task_name: str, params: dict):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context: OllamaContext = context
        self.__task_name: str = task_name
        self.__params: dict = params

    async def setup(self):
        self.__logger.info(f"Setting up DocumentIndexing task {self.__task_name} with params {self.__params}...")

    async def run(self):
        # Start all tasks
        self.__logger.debug(f"Running task processor {self.__task_name}")

        while not self.__context.stopping:
            self.__logger.debug(f"Still running task {self.__task_name}")
            await self.__context.wait(30)

    async def tick(self):
        """
        Gets called whenever this ollama_relai instance wins a tick from ceduleur (so once a minute for all ollama_relai instances)
        :return:
        """
        pass

    async def producer_task(self):
        pass
