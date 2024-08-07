import logging

from millegrilles_messages.MilleGrillesConnecteur import EtatInstance
from millegrilles_ollama_relai.Configuration import Configuration as ConfigurationOllamaRelai


class OllamaRelaiContext(EtatInstance):

    def __init__(self, configuration: ConfigurationOllamaRelai):
        super().__init__(configuration)
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    async def reload_configuration(self):
        await super().reload_configuration()

    @property
    def configuration(self) -> ConfigurationOllamaRelai:
        return super().configuration

