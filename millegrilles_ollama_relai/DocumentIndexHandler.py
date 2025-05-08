import logging

from millegrilles_ollama_relai.OllamaContext import OllamaContext


class DocumentIndexHandler:

    def __init__(self, context: OllamaContext):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context

