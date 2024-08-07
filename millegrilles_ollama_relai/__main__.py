import asyncio
import logging

from millegrilles_ollama_relai.OllamaRelai import run
from millegrilles_ollama_relai.Configuration import parse_args


def main():
    """
    :return:
    """
    logging_setup()
    configuration = parse_args()
    asyncio.run(run(configuration))


def logging_setup():
    logging.basicConfig()
    logging.getLogger(__name__).setLevel(logging.WARN)
    logging.getLogger('millegrilles_messages').setLevel(logging.WARN)
    logging.getLogger('millegrilles_ollama_relai').setLevel(logging.WARN)


if __name__ == '__main__':
    main()
