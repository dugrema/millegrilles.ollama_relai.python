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
from millegrilles_ollama_relai.OllamaInstanceManager import OllamaInstanceManager, OllamaInstance
from millegrilles_ollama_relai.OllamaManager import OllamaManager
from millegrilles_ollama_relai import Constantes as OllamaConstants


class MgbusHandler:
    """
    MQ access module
    """

    def __init__(self, manager: OllamaManager, ollama_instances: OllamaInstanceManager):
        super().__init__()
        self.__logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
        self.__manager = manager
        self.__ollama_instances = ollama_instances
        self.__task_group: Optional[TaskGroup] = None

        # Wire message processing callback
        ollama_instances.set_message_cb(self.__on_ollama_instance_message)

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

        channel_triggers = create_trigger_q_channel(context, self.__on_processing_trigger)
        await self.__manager.context.bus_connector.add_channel(channel_triggers)

    async def __on_volatile_message(self, message: MessageWrapper):
        # Authorization check
        enveloppe = message.certificat
        try:
            roles = enveloppe.get_roles
        except ExtensionNotFound:
            roles = list()
        try:
            domains_env = enveloppe.get_domaines
        except ExtensionNotFound:
            domains_env = None

        message_type = message.routing_key.split('.')[0]
        domain = message.routage['domaine']
        action = message.routage['action']
        estampille = message.estampille

        # Volatile messages expire after 90 seconds
        expired_timestamp = (datetime.datetime.now() - datetime.timedelta(seconds=90)).timestamp()
        if estampille < expired_timestamp:
            return None  # Ignore

        if message_type == 'evenement':
            if domain == 'filecontroler' and action == 'filehostNewFuuid' and 'filecontroler' in roles:
                # Create delayed task - we are listening to the filecontroler, need to give time for GrosFichiers to
                # register the visit
                asyncio.create_task(self.__manager.trigger_rag_indexing(delay=3))
                return None
            elif domain == 'AiLanguage' and action == 'configurationUpdated' and 'AiLanguage' in domains_env:
                await self.__manager.trigger_reload_ai_configuration()
                return None

        if Constantes.ROLE_USAGER not in roles:
            return {'ok': False, 'code': 403, 'err': 'Acces denied'}

        if message_type == 'requete':
            if action == 'ping':
                ready = self.__ollama_instances.ready
                return {'ok': ready}
            elif action == 'getModels':
                models = self.__ollama_instances.get_models()
                return {'ok': True, 'models': models}
            elif action == 'queryRag':
                return await self.__manager.register_rag_query(message)
        elif message_type == 'commande':
            if action in ['chat', 'knowledge_query']:
                return await self.__manager.register_chat(message)

        self.__logger.info("__on_volatile_message Ignoring unknown action %s", message.routing_key)
        return {'ok': False, 'code': 404, 'err': 'Unknown operation'}

    async def __on_ollama_instance_message(self, instance: OllamaInstance, message: MessageWrapper):
        """
        Called for processing a message from an instance
        :param instance: Instance that received the message from its own work queue.
        :param message: Message to process
        :return:
        """
        # Authorization check
        enveloppe = message.certificat
        try:
            roles = enveloppe.get_roles
        except ExtensionNotFound:
            roles = list()

        if Constantes.ROLE_USAGER not in roles:
            return {'ok': False, 'code': 403, 'err': 'Acces denied'}

        action = message.routage['action']

        # Dedupe messages received on instance work queues with same models
        # Ollama instances are registered for all their models. A processing message goes to all supporting queues at
        # the same time. This is used to ensure only one ollama instance is allowed to process the same message by id.
        message_id = message.id
        try:
            await self.__ollama_instances.claim_query(message_id)
        except Exception:
            self.__logger.debug(f"Query {message_id} already running on other ollama instance, skipping")
            return None

        if action in ['chat', 'knowledge_query']:
            return await self.__manager.process_chat(instance, message)
        elif action == 'queryRag':
            return await self.__manager.query_rag(instance, message)

        self.__logger.info("__on_processing_message Ignoring unknown action %s", message.routing_key)
        return {'ok': False, 'code': 404, 'err': 'Unknown operation'}

    async def __on_processing_trigger(self, message: MessageWrapper):
        # Authorization check
        enveloppe = message.certificat
        try:
            roles = enveloppe.get_roles
        except ExtensionNotFound:
            roles = list()

        if Constantes.ROLE_USAGER not in roles:
            return {'ok': False, 'code': 403, 'err': 'Acces denied'}

        action = message.routage['action']

        if action == 'cancelChat':
            return await self.__manager.cancel_chat(message)

        self.__logger.info("__on_processing_message Ignoring unknown action %s", message.routing_key)
        return {'ok': False, 'code': 404, 'err': 'Unknown operation'}


def create_volatile_q_channel(context: MilleGrillesBusContext,
                               on_message: Callable[[MessageWrapper], Coroutine[Any, Any, None]]) -> MilleGrillesPikaChannel:

    q_channel = MilleGrillesPikaChannel(context, prefetch_count=1)
    q_instance = MilleGrillesPikaQueueConsumer(
        context, on_message, 'ollama_relai/volatile', arguments={'x-message-ttl': 30_000})

    q_instance.add_routing_key(RoutingKey(
        Constantes.SECURITE_PROTEGE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.pull'))
    q_instance.add_routing_key(RoutingKey(
        Constantes.SECURITE_PRIVE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.generate'))
    q_instance.add_routing_key(RoutingKey(
        Constantes.SECURITE_PRIVE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.chat'))
    q_instance.add_routing_key(RoutingKey(
        Constantes.SECURITE_PRIVE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.knowledge_query'))
    q_instance.add_routing_key(RoutingKey(
        Constantes.SECURITE_PRIVE, f'requete.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.ping'))
    q_instance.add_routing_key(RoutingKey(
        Constantes.SECURITE_PRIVE, f'requete.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.getModels'))

    q_instance.add_routing_key(RoutingKey(
        Constantes.SECURITE_PRIVE, 'evenement.AiLanguage.configurationUpdated'))

    if context.configuration.rag_active or context.configuration.summary_active:  # RAG (document index)
        q_instance.add_routing_key(RoutingKey(
            Constantes.SECURITE_PUBLIC, 'evenement.filecontroler.filehostNewFuuid'))

    if context.configuration.rag_active:  # RAG (document index)
        q_instance.add_routing_key(RoutingKey(
            Constantes.SECURITE_PROTEGE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.indexDocuments'))
        q_instance.add_routing_key(RoutingKey(
            Constantes.SECURITE_PRIVE, f'requete.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.queryRag'))

    q_channel.add_queue(q_instance)
    return q_channel


def create_trigger_q_channel(context: MilleGrillesBusContext, on_message: Callable[[MessageWrapper], Coroutine[Any, Any, None]]) -> MilleGrillesPikaChannel:
    # System triggers
    trigger_q_channel = MilleGrillesPikaChannel(context, prefetch_count=1)
    trigger_q = MilleGrillesPikaQueueConsumer(context, on_message, exclusive=True, arguments={'x-message-ttl': 30_000})
    trigger_q_channel.add_queue(trigger_q)
    trigger_q.add_routing_key(RoutingKey(Constantes.SECURITE_PRIVE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.cancelChat'))

    return trigger_q_channel
