import asyncio
import datetime
import logging

from asyncio import TaskGroup
from typing import Optional, Callable, Coroutine, Any

from cryptography.x509 import ExtensionNotFound

from millegrilles_messages.bus.BusContext import MilleGrillesBusContext, ForceTerminateExecution
from millegrilles_messages.messages import Constantes
from millegrilles_messages.bus.PikaChannel import MilleGrillesPikaChannel
from millegrilles_messages.bus.PikaQueue import MilleGrillesPikaQueueConsumer, RoutingKey
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.OllamaManager import OllamaManager


class MgbusHandler:
    """
    MQ access module
    """

    def __init__(self, manager: OllamaManager):
        super().__init__()
        self.__logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
        self.__manager = manager
        self.__task_group: Optional[TaskGroup] = None

    async def run(self):
        self.__logger.debug("MgbusHandler thread started")
        try:
            await self.__register()

            async with TaskGroup() as group:
                self.__task_group = group
                group.create_task(self.__stop_thread())
                group.create_task(self.__manager.context.bus_connector.run())

        except *Exception:  # Stop on any thread exception
            if self.__manager.context.stopping is False:
                self.__logger.exception("GenerateurCertificatsHandler Unhandled error, closing")
                self.__manager.context.stop()
                raise ForceTerminateExecution()
        self.__task_group = None
        self.__logger.debug("MgbusHandler thread done")

    async def __stop_thread(self):
        await self.__manager.context.wait()

    async def __register(self):
        self.__logger.info("Register with the MQ Bus")

        context = self.__manager.context

        channel_volatile = create_volatile_q_channel(context, self.__on_volatile_message)
        await self.__manager.context.bus_connector.add_channel(channel_volatile)

    async def __on_volatile_message(self, message: MessageWrapper):
        # Authorization check
        enveloppe = message.certificat
        try:
            roles = enveloppe.get_roles
        except ExtensionNotFound:
            roles = list()

        message_type = message.routing_key.split('.')[0]
        domain = message.routage['domaine']
        action = message.routage['action']
        estampille = message.estampille

        # Volatile messages expire after 90 seconds
        expired_timestamp = (datetime.datetime.now() - datetime.timedelta(seconds=90)).timestamp()
        if estampille < expired_timestamp:
            return None  # Ignore

        if message_type == 'evenement':
            if domain == 'CoreTopologie' and action == 'configurationFile' and message.parsed.get('filename') == 'ollama_relai':
                self.__logger.info("Received configuration update trigger")
                await self.__manager.trigger_reload_ai_configuration()
                return None

        if Constantes.ROLE_USAGER not in roles:
            return {'ok': False, 'code': 403, 'err': 'Acces denied'}

        self.__logger.info("__on_volatile_message Ignoring unknown action %s", message.routing_key)
        return {'ok': False, 'code': 404, 'err': 'Unknown operation'}


def create_volatile_q_channel(context: MilleGrillesBusContext,
                               on_message: Callable[[MessageWrapper], Coroutine[Any, Any, None]]) -> MilleGrillesPikaChannel:

    q_channel = MilleGrillesPikaChannel(context, prefetch_count=1)
    q_instance = MilleGrillesPikaQueueConsumer(
        context, on_message, 'ollama_relai/volatile', arguments={'x-message-ttl': 30_000})

    q_instance.add_routing_key(RoutingKey(
        Constantes.SECURITE_PUBLIC, 'evenement.CoreTopologie.configurationFile'))

    q_channel.add_queue(q_instance)
    return q_channel
