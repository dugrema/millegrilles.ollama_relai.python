import argparse
import logging

from millegrilles_messages.MilleGrillesConnecteur import Configuration as ConfigurationConnecteur


CONST_PARAMS = []


class Configuration(ConfigurationConnecteur):

    def __init__(self):
        super().__init__()
        logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

    def get_params_list(self) -> list:
        params = super().get_params_list()
        params.extend(CONST_PARAMS)
        return params


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
