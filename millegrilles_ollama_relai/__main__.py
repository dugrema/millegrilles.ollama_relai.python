import asyncio
import logging
from asyncio import TaskGroup
from concurrent.futures.thread import ThreadPoolExecutor

from typing import Awaitable

from millegrilles_messages.bus.BusContext import ForceTerminateExecution, StopListener
from millegrilles_messages.bus.PikaConnector import MilleGrillesPikaConnector
from millegrilles_ollama_relai.AttachmentHandler import AttachmentHandler
from millegrilles_ollama_relai.MgbusHandler import MgbusHandler
from millegrilles_ollama_relai.OllamaChatHandler import OllamaChatHandler
from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.OllamaManager import OllamaManager
from millegrilles_ollama_relai.QueryHandler import QueryHandler

LOGGER = logging.getLogger(__name__)


async def force_terminate_task_group():
    """Used to force termination of a task group."""
    raise ForceTerminateExecution()


async def main():
    config = OllamaConfiguration.load()
    context = OllamaContext(config)

    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Starting")

    # Wire classes together, gets awaitables to run
    coros = await wiring(context)

    try:
        # Use taskgroup to run all threads
        async with TaskGroup() as group:
            for coro in coros:
                group.create_task(coro)

            # Create a listener that fires a task to cancel all other tasks
            async def stop_group():
                group.create_task(force_terminate_task_group())
            stop_listener = StopListener(stop_group)
            context.register_stop_listener(stop_listener)

    except* (ForceTerminateExecution, asyncio.CancelledError):
        pass  # Result of the termination task


async def wiring(context: OllamaContext) -> list[Awaitable]:
    # Some threads get used to handle sync events for the duration of the execution. Ensure there are enough.
    loop = asyncio.get_event_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=10))

    # Service instances
    bus_connector = MilleGrillesPikaConnector(context)
    context.bus_connector = bus_connector
    attachment_handler = AttachmentHandler(context)
    chat_handler = OllamaChatHandler(context, attachment_handler)
    query_handler = QueryHandler(context, chat_handler)

    # Facade
    manager = OllamaManager(context, query_handler, attachment_handler, chat_handler)

    # Access modules
    bus_handler = MgbusHandler(manager)

    # Setup, injecting additional dependencies
    await manager.setup()  # Create folders for other modules

    # Create tasks
    coros = [
        context.run(),
        query_handler.run(),
        manager.run(),
        bus_handler.run(),
        chat_handler.run(),
    ]

    return coros


if __name__ == '__main__':
    asyncio.run(main())
    LOGGER.info("Stopped")


# import argparse
# import asyncio
# import logging
#
# from millegrilles_ollama_relai.OllamaRelai import run
# from millegrilles_ollama_relai.Configuration import parse_args
# from millegrilles_ollama_relai.Context import OllamaRelaiContext
#
#
# def main():
#     """
#     :return:
#     """
#     logging_setup()
#
#     # Parse command line, load configuration and create context object
#     configuration, args = parse_args()
#     adjust_logging(args)
#     context = OllamaRelaiContext(configuration)
#
#     # Run
#     asyncio.run(run(context))
#
#
# def logging_setup():
#     logging.basicConfig(level=logging.ERROR)
#     logging.getLogger(__name__).setLevel(logging.WARN)
#     logging.getLogger('millegrilles_messages').setLevel(logging.WARN)
#     logging.getLogger('millegrilles_ollama_relai').setLevel(logging.INFO)
#
#
# def adjust_logging(args: argparse.Namespace):
#     if args.verbose:
#         # Set millegrilles modules logging to verbose
#         logging.getLogger('millegrilles_messages').setLevel(logging.DEBUG)
#         logging.getLogger('millegrilles_ollama_relai').setLevel(logging.DEBUG)
#
#         # Confirmation
#         logging.getLogger('millegrilles_ollama_relai').debug("** Verbose logging **")
#
#
# if __name__ == '__main__':
#     main()
