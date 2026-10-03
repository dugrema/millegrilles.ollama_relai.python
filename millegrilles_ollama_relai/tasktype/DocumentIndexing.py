from millegrilles_ollama_relai.OllamaContext import OllamaContext


class Processor:

    def __init__(self, context: OllamaContext, params: dict):
        self.__context: OllamaContext = context

    async def run(self):
        # Start all tasks


        while not self.__context.stopping:
            # TDOD
            await self.__context.wait(30)

    async def tick(self):
        """
        Gets called whenever this ollama_relai instance wins a tick from ceduleur (so once a minute for all ollama_relai instances)
        :return:
        """
        pass

    async def producer_task(self):
        pass
