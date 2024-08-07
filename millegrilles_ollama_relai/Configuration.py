import argparse
import logging

from typing import Optional

from millegrilles_messages.MilleGrillesConnecteur import Configuration as ConfigurationConnecteur
from millegrilles_ollama_relai import Constantes


CONST_PARAMS = [
    Constantes.ENV_OLLAMA_URL,
]


class Configuration(ConfigurationConnecteur):

    def __init__(self):
        super().__init__()
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.ollama_url = Constantes.DEFAULT_OLLAMA_URL

    def get_params_list(self) -> list:
        params = super().get_params_list()
        params.extend(CONST_PARAMS)
        return params

    def parse_config(self, configuration: Optional[dict] = None) -> dict:
        dict_params = super().parse_config(configuration)

        # Parametres optionnels / overrides
        self.ollama_url = dict_params.get(Constantes.ENV_OLLAMA_URL) or self.ollama_url

        return dict_params


def parse_args() -> (Configuration, argparse.Namespace):
    parser = argparse.ArgumentParser(description="Relay between the MilleGrilles MQ bus and Ollama instances")

    parser.add_argument(
        '--verbose', action="store_true", required=False,
        help="Activates verbose logging"
    )

    args = parser.parse_args()

    # Prepare the configuration instance
    config = Configuration()
    config.parse_config(args.__dict__)

    return config, args
