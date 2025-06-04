import asyncio
import logging
from asyncio import TaskGroup

from millegrilles_messages.messages import Constantes
from millegrilles_messages.bus.BusContext import ForceTerminateExecution, StopListener
from millegrilles_messages.bus.PikaConnector import MilleGrillesPikaConnector
from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration
from millegrilles_ollama_relai.OllamaContext import OllamaContext

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


async def test_fn(context: OllamaContext):
    producer = await context.get_producer()
    LOGGER.info("READY")

    response = await producer.request({}, 'AiLanguage', 'Test', exchange=Constantes.SECURITE_PRIVE)

    LOGGER.info("DONE")


# ****************
# Scaffolding code
# ****************
async def force_terminate_task_group():
    """Used to force termination of a task group."""
    raise ForceTerminateExecution()


async def main():
    config = OllamaConfiguration.load()
    context = OllamaContext(config)

    LOGGER.setLevel(logging.INFO)
    LOGGER.info("Starting")

    # Wire classes together, gets awaitables to run
    # coros = await wiring(context)
    bus_connector = MilleGrillesPikaConnector(context)
    context.bus_connector = bus_connector

    async def run_test():
        await test_fn(context)
        context.stop()

    try:
        # Use taskgroup to run all threads
        async with TaskGroup() as group:
            group.create_task(context.run())
            group.create_task(context.bus_connector.run())
            group.create_task(run_test())

            # Create a listener that fires a task to cancel all other tasks
            async def stop_group():
                group.create_task(force_terminate_task_group())
            stop_listener = StopListener(stop_group)
            context.register_stop_listener(stop_listener)

    except* (ForceTerminateExecution, asyncio.CancelledError):
        pass  # Result of the termination task


if __name__ == '__main__':
    asyncio.run(main())
    LOGGER.info("Stopped")
