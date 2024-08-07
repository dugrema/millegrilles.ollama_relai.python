import argparse
import asyncio
import logging

from millegrilles_ollama_relai.OllamaRelai import run
from millegrilles_ollama_relai.Configuration import parse_args
from millegrilles_ollama_relai.Context import OllamaRelaiContext


def main():
    """
    :return:
    """
    logging_setup()

    # Parse command line, load configuration and create context object
    configuration, args = parse_args()
    adjust_logging(args)
    context = OllamaRelaiContext(configuration)

    # Run
    asyncio.run(run(context))


def logging_setup():
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(__name__).setLevel(logging.WARN)
    logging.getLogger('millegrilles_messages').setLevel(logging.WARN)
    logging.getLogger('millegrilles_ollama_relai').setLevel(logging.INFO)


def adjust_logging(args: argparse.Namespace):
    if args.verbose:
        # Set millegrilles modules logging to verbose
        logging.getLogger('millegrilles_messages').setLevel(logging.DEBUG)
        logging.getLogger('millegrilles_ollama_relai').setLevel(logging.DEBUG)

        # Confirmation
        logging.getLogger('millegrilles_ollama_relai').debug("** Verbose logging **")


if __name__ == '__main__':
    main()
