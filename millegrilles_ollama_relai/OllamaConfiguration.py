import argparse
import logging

from os import environ

from millegrilles_messages.bus.BusConfiguration import MilleGrillesBusConfiguration
from millegrilles_messages.messages import Constantes as Constantes
from millegrilles_ollama_relai import Constantes as OllamaConstants

LOGGING_NAMES = [__name__, 'millegrilles_messages', 'millegrilles_ollama_relai']


def __adjust_logging(args: argparse.Namespace):
    logging.basicConfig()
    if args.verbose is True:
        for log in LOGGING_NAMES:
            logging.getLogger(log).setLevel(logging.DEBUG)
    else:
        for log in LOGGING_NAMES:
            logging.getLogger(log).setLevel(logging.INFO)


def _parse_command_line():
    parser = argparse.ArgumentParser(description="Relay between the MilleGrilles MQ bus and Ollama instances")
    parser.add_argument(
        '--verbose', action="store_true", required=False,
        help="More logging"
    )

    parser.add_argument(
        '--rag', action="store_true", required=False,
        help="Activate RAG (Resource Augmented Generation) with persistent local vector database"
    )

    args = parser.parse_args()
    __adjust_logging(args)
    return args



class OllamaConfiguration(MilleGrillesBusConfiguration):

    def __init__(self):
        super().__init__()
        self.dir_staging = '/var/opt/millegrilles/staging'
        self.dir_rag = '/var/opt/millegrilles/rag'
        self.ollama_url = OllamaConstants.DEFAULT_OLLAMA_URL
        self.rag_active = False

    def parse_config(self):
        super().parse_config()

        self.dir_staging = environ.get(Constantes.ENV_DIR_STAGING) or self.dir_staging
        self.dir_rag = environ.get(OllamaConstants.ENV_DIR_AI_RAG) or self.dir_rag
        self.ollama_url = environ.get(OllamaConstants.ENV_OLLAMA_URL) or self.ollama_url

    def __set_args(self, args: argparse.Namespace):
        self.rag_active = args.rag or False

    @staticmethod
    def load():
        # Override
        config = OllamaConfiguration()
        args = _parse_command_line()
        config.parse_config()
        config.__set_args(args)
        return config
