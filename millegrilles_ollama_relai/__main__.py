import asyncio
import logging
from asyncio import TaskGroup
from concurrent.futures.thread import ThreadPoolExecutor

from typing import Awaitable

from millegrilles_messages.bus.BusContext import ForceTerminateExecution, StopListener
from millegrilles_messages.bus.PikaConnector import MilleGrillesPikaConnector
from millegrilles_messages.Filehost import FilehostConnection
from millegrilles_ollama_relai.DocumentIndexHandler import DocumentIndexHandler
from millegrilles_ollama_relai.MgbusHandler import MgbusHandler
from millegrilles_ollama_relai.OllamaChatHandler import OllamaChatHandler
from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.OllamaInstanceManager import OllamaInstanceManager
from millegrilles_ollama_relai.OllamaManager import OllamaManager
from millegrilles_ollama_relai.OllamaTools import OllamaToolHandler

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
    ollama_instances = OllamaInstanceManager(context)
    attachment_handler = FilehostConnection(context)
    tool_handler = OllamaToolHandler(context, attachment_handler)
    chat_handler = OllamaChatHandler(context, ollama_instances, attachment_handler, tool_handler)
    document_handler = DocumentIndexHandler(context, ollama_instances, attachment_handler)

    # Facade
    manager = OllamaManager(context, ollama_instances, attachment_handler, tool_handler, chat_handler, document_handler)

    # Access modules
    bus_handler = MgbusHandler(manager, ollama_instances)

    # Setup, injecting additional dependencies
    await manager.setup()  # Create folders for other modules
    await document_handler.setup()  # Connect to local vector DB
    await tool_handler.setup()

    # Create tasks
    coros = [
        context.run(),
        ollama_instances.run(),
        manager.run(),
        bus_handler.run(),
        tool_handler.run(),
        chat_handler.run(),
        document_handler.run(),
    ]

    return coros


if __name__ == '__main__':
    asyncio.run(main())
    LOGGER.info("Stopped")
