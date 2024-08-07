import logging

from typing import Optional
from cryptography.x509.extensions import ExtensionNotFound

from millegrilles_messages.messages import Constantes as ConstantesMilleGrilles
from millegrilles_messages.messages.MessagesThread import MessagesThread
from millegrilles_messages.MilleGrillesConnecteur import CommandHandler as CommandesAbstract, RoutingKey
from millegrilles_messages.messages.MessagesModule import MessageProducerFormatteur, MessageWrapper, RessourcesConsommation
from millegrilles_ollama_relai.QueryHandler import QueryHandler


from millegrilles_ollama_relai.Context import OllamaRelaiContext


class CommandHandler(CommandesAbstract):

    def __init__(self, relai_app, query_handler: QueryHandler):
        super().__init__()
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__relai_app = relai_app
        self.__messages_thread: Optional[MessagesThread] = None
        self.__query_handler = query_handler

    @property
    def context(self) -> OllamaRelaiContext:
        return self.__relai_app.context

    def get_routing_keys(self):
        # Ecouter les evenements sur les commandes. Permet a plusieurs relais de coordonner les executions.
        return [
            RoutingKey('evenement.ollama_relai.*.debutTraitement', ConstantesMilleGrilles.SECURITE_PRIVE),
            RoutingKey('evenement.ollama_relai.*.miseajour', ConstantesMilleGrilles.SECURITE_PRIVE),
            RoutingKey('evenement.ollama_relai.*.resultat', ConstantesMilleGrilles.SECURITE_PRIVE),
            RoutingKey('evenement.ollama_relai.*.annule', ConstantesMilleGrilles.SECURITE_PRIVE),
        ]

    def configurer_consumers(self, messages_thread: MessagesThread):
        self.__messages_thread = messages_thread

        nom_ou = self.context.clecertificat.enveloppe.subject_organizational_unit_name
        if nom_ou is None:
            raise Exception('Invalid certificate - no Organizational Unit (OU) name')

        self.configurer_q_volatil(nom_ou)
        self.configurer_q_traitement(nom_ou)

    def configurer_q_volatil(self, nom_ou: str):
        # Creer une Q pour les messages volatils. Utilise le nom du OU dans le certificat.
        # Les messages sont traites rapidement (pas de longue duree d'execution).
        nom_queue_ou = f'{nom_ou}/volatil'
        res_volatil = RessourcesConsommation(
            self.callback_reply_q, nom_queue=nom_queue_ou, channel_separe=True, est_asyncio=True, durable=True)
        res_volatil.set_ttl(300_000)  # millisecs
        res_volatil.ajouter_rk(
            ConstantesMilleGrilles.SECURITE_PUBLIC,
            f'evenement.{ConstantesMilleGrilles.ROLE_CEDULEUR}.{ConstantesMilleGrilles.EVENEMENT_PING_CEDULE}')

        res_volatil.ajouter_rk(ConstantesMilleGrilles.SECURITE_PROTEGE, 'commande.ollama_relai.pull')
        res_volatil.ajouter_rk(ConstantesMilleGrilles.SECURITE_PRIVE, 'commande.ollama_relai.generate')
        res_volatil.ajouter_rk(ConstantesMilleGrilles.SECURITE_PRIVE, 'commande.ollama_relai.chat')

        self.__messages_thread.ajouter_consumer(res_volatil)

    def configurer_q_traitement(self, nom_ou: str):
        # Q de traitement. Toutes les commandes sont transferees vers cette Q apres confirmation initiale
        nom_queue_ou = f'{nom_ou}/traitement'
        res_traitement = RessourcesConsommation(
            self.callback_reply_q, nom_queue=nom_queue_ou, channel_separe=True, est_asyncio=True, durable=True)
        res_traitement.set_ttl(900_000)  # millisecs

        res_traitement.ajouter_rk(ConstantesMilleGrilles.SECURITE_PROTEGE, 'commande.ollama_relai.traitement')

        self.__messages_thread.ajouter_consumer(res_traitement)

    async def traiter_commande(self, producer: MessageProducerFormatteur, message: MessageWrapper):
        routing_key = message.routing_key
        exchange = message.exchange
        q = message.queue

        rk_split = routing_key.split('.')
        type_message = rk_split[0]
        domaine = rk_split[1]
        action = rk_split.pop()
        enveloppe = message.certificat

        try:
            exchanges = enveloppe.get_exchanges
        except ExtensionNotFound:
            exchanges = list()

        try:
            roles = enveloppe.get_roles
        except ExtensionNotFound:
            roles = list()

        try:
            user_id = enveloppe.get_user_id
        except ExtensionNotFound:
            user_id = list()

        try:
            delegation_globale = enveloppe.get_delegation_globale
        except ExtensionNotFound:
            delegation_globale = None

        if q == 'ollama_relai/traitement':
            return await self.__query_handler.process_query(message)

        if domaine == 'ollama_relai':
            return await self.__query_handler.handle_query(message)
