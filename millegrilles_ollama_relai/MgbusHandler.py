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
from millegrilles_ollama_relai import Constantes as OllamaConstants


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

        channel_processing = create_processing_q_channel(context, self.__on_processing_message)
        await self.__manager.context.bus_connector.add_channel(channel_processing)

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

        if message_type == 'evenement':
            if domain == 'filecontroler' and action == 'filehostNewFuuid' and 'filecontroler' in roles:
                await self.__manager.trigger_rag_indexing()
                return None
            elif domain == 'AiLanguage' and action == 'configurationUpdated' and 'AiLanguage' in domains_env:
                await self.__manager.trigger_reload_ai_configuration()
                return None

        if Constantes.ROLE_USAGER not in roles:
            return {'ok': False, 'code': 403, 'err': 'Acces denied'}

        if action in ['chat']:
            return await self.__manager.handle_volalile_query(message)
        elif action in ['pull', 'ping', 'getModels', 'queryRag']:
            return await self.__manager.handle_volalile_request(message)

        self.__logger.info("__on_volatile_message Ignoring unknown action %s", message.routing_key)
        return {'ok': False, 'code': 404, 'err': 'Unknown operation'}

    async def __on_processing_message(self, message: MessageWrapper):
        # Authorization check
        enveloppe = message.certificat
        try:
            roles = enveloppe.get_roles
        except ExtensionNotFound:
            roles = list()

        if Constantes.ROLE_USAGER not in roles:
            return {'ok': False, 'code': 403, 'err': 'Acces denied'}

        action = message.routage['action']

        if action == 'chat':
            return await self.__manager.process_chat(message)

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
        Constantes.SECURITE_PRIVE, f'requete.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.ping'))
    q_instance.add_routing_key(RoutingKey(
        Constantes.SECURITE_PRIVE, f'requete.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.getModels'))

    if context.configuration.rag_active:  # RAG (document index)
        q_instance.add_routing_key(RoutingKey(
            Constantes.SECURITE_PROTEGE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.indexDocuments'))
        q_instance.add_routing_key(RoutingKey(
            Constantes.SECURITE_PRIVE, f'requete.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.queryRag'))
        q_instance.add_routing_key(RoutingKey(
            Constantes.SECURITE_PUBLIC, 'evenement.filecontroler.filehostNewFuuid'))
        q_instance.add_routing_key(RoutingKey(
            Constantes.SECURITE_PRIVE, 'evenement.AiLanguage.configurationUpdated'))

    q_channel.add_queue(q_instance)
    return q_channel


def create_processing_q_channel(context: MilleGrillesBusContext,
                               on_message: Callable[[MessageWrapper], Coroutine[Any, Any, None]]) -> MilleGrillesPikaChannel:

    nom_ou = context.signing_key.enveloppe.subject_organizational_unit_name
    if nom_ou is None:
        raise Exception('Invalid certificate - no Organizational Unit (OU) name')
    queue_name = f'{nom_ou}/processing'

    q_channel = MilleGrillesPikaChannel(context, prefetch_count=1)
    q_instance = MilleGrillesPikaQueueConsumer(context, on_message, queue_name, arguments={'x-message-ttl': 900_000})

    q_instance.add_routing_key(RoutingKey(
        Constantes.SECURITE_PROTEGE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.traitement'))

    q_channel.add_queue(q_instance)
    return q_channel
