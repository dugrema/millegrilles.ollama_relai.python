import asyncio
import logging
from asyncio import TaskGroup
from typing import Callable, Any, Optional, Coroutine

from millegrilles_messages.bus.PikaChannel import MilleGrillesPikaChannel
from millegrilles_messages.bus.PikaQueue import MilleGrillesPikaQueueConsumer, RoutingKey
from millegrilles_messages.messages import Constantes
from millegrilles_messages.bus.BusContext import ForceTerminateExecution, StopListener, MilleGrillesBusContext
from millegrilles_messages.bus.PikaConnector import MilleGrillesPikaConnector
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai import Constantes as OllamaConstants

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def create_volatile_q_channel(context: MilleGrillesBusContext,
                              on_message: Callable[[MessageWrapper], Coroutine[Any, Any, None]]) -> MilleGrillesPikaChannel:

    q_channel = MilleGrillesPikaChannel(context, prefetch_count=1)

    q_instance_dispatch = MilleGrillesPikaQueueConsumer(
        context, on_message, 'ollama_relai/testdispatch', arguments={'x-message-ttl': 180_000})
    q_instance_dispatch.add_routing_key(RoutingKey(
        Constantes.SECURITE_PRIVE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.dispatch'))

    q_instance_1 = MilleGrillesPikaQueueConsumer(
        context, on_message, 'ollama_relai/test1', arguments={'x-message-ttl': 180_000})
    q_instance_1.add_routing_key(RoutingKey(
        Constantes.SECURITE_PRIVE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.model1'))

    q_instance_2 = MilleGrillesPikaQueueConsumer(
        context, on_message, 'ollama_relai/test2', arguments={'x-message-ttl': 180_000})
    q_instance_2.add_routing_key(RoutingKey(
        Constantes.SECURITE_PRIVE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.model2'))

    q_channel.add_queue(q_instance_dispatch)
    q_channel.add_queue(q_instance_1)
    q_channel.add_queue(q_instance_2)

    return q_channel


class MessageHandler:

    def __init__(self, context: OllamaContext):
        self.context = context

    async def message_received(self, message: MessageWrapper):
        action = message.routing_key.split('.')[-1]
        if action == 'dispatch':
            print(f"Message received, dispatching to {message.parsed['model']}")
            producer = await self.context.get_producer()
            action = f"model{message.parsed['model']}"
            await producer.command(message.original, OllamaConstants.DOMAIN_OLLAMA_RELAI, action, exchange=Constantes.SECURITE_PRIVE,
                                   nowait=True, noformat=True)
            return {'ok': True}
        else:
            print(f"Message received: {message.routage}: {message.parsed['no']}")
            await asyncio.sleep(3)
            print(f"Processing done: {message.routage}: {message.parsed['no']}")

            if action == 'model1':
                pass
            elif action == 'model2':
                pass

        return None


async def setup(context: OllamaContext) -> list[MilleGrillesPikaChannel]:
    message_handler = MessageHandler(context)
    channel1 = create_volatile_q_channel(context, message_handler.message_received)
    return [channel1]


async def test_fn(context: OllamaContext):
    producer = await context.get_producer()
    LOGGER.info("READY")

    await asyncio.sleep(1)  # Await Q creation

    for i in range(1, 3):
        await producer.command({"no": i, "model": 1}, OllamaConstants.DOMAIN_OLLAMA_RELAI, 'dispatch', exchange=Constantes.SECURITE_PRIVE,
                               nowait=True)
        await producer.command({"no": i, "model": 2}, OllamaConstants.DOMAIN_OLLAMA_RELAI, 'dispatch', exchange=Constantes.SECURITE_PRIVE,
                               nowait=True)

    await asyncio.sleep(30)

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

    channels = await setup(context)
    for channel in channels:
        await bus_connector.add_channel(channel)

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
