import logging

from typing import Optional

from millegrilles_messages.messages import Constantes as ConstantesMilleGrilles
from millegrilles_messages.messages.MessagesThread import MessagesThread
from millegrilles_messages.MilleGrillesConnecteur import CommandHandler as CommandesAbstract
from millegrilles_messages.messages.MessagesModule import MessageProducerFormatteur, MessageWrapper, RessourcesConsommation
from millegrilles_ollama_relai.Context import OllamaRelaiContext


class CommandHandler(CommandesAbstract):

    def __init__(self, relai_app):
        super().__init__()
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__relai_app = relai_app
        self.__messages_thread: Optional[MessagesThread] = None

    @property
    def context(self) -> OllamaRelaiContext:
        return self.__relai_app.context

    def get_routing_keys(self):
        return [
            # f'evenement.{Constantes.DOMAINE_GROSFICHIERS}.{Constantes.EVENEMENT_GROSFICHIERS_CHANGEMENT_CONSIGNATION_PRIMAIRE}',
            # 'evenement.CoreTopologie.changementConsignation',
        ]

    def configurer_consumers(self, messages_thread: MessagesThread):
        self.__messages_thread = messages_thread
        # Creer une Q pour les messages volatils. Utilise le nom du OU dans le certificat.
        nom_ou = self.context.clecertificat.enveloppe.subject_organizational_unit_name
        if nom_ou is not None:
            nom_queue_ou = f'{nom_ou}/volatil'
            res_volatil = RessourcesConsommation(self.callback_reply_q, nom_queue=nom_queue_ou, channel_separe=True,
                                                 est_asyncio=True, durable=True)
            res_volatil.set_ttl(300000)  # millisecs
            res_volatil.ajouter_rk(
                ConstantesMilleGrilles.SECURITE_PUBLIC,
                f'evenement.{ConstantesMilleGrilles.ROLE_CEDULEUR}.{ConstantesMilleGrilles.EVENEMENT_PING_CEDULE}', )
            messages_thread.ajouter_consumer(res_volatil)
        else:
            raise Exception('Invalid certificate - no Organizational Unit (OU) name')

        # res_evenements = RessourcesConsommation(self.callback_reply_q, channel_separe=True, est_asyncio=True)
        #
        # res_evenements.ajouter_rk(
        #     ConstantesMilleGrilles.SECURITE_PUBLIC,
        #     f'evenement.{ConstantesMilleGrilles.DOMAINE_MAITRE_DES_CLES}.{ConstantesMilleGrilles.EVENEMENT_MAITREDESCLES_CERTIFICAT}', )
        #
        # # Listener pour les subscriptions. Les routing keys sont gerees par subsbscription_handler dynamiquement.
        # res_subscriptions = RessourcesConsommation(
        #     self.socket_io_handler.subscription_handler.callback_reply_q,
        #     channel_separe=True, est_asyncio=True)
        # self.socket_io_handler.subscription_handler.messages_thread = messages_thread
        # self.socket_io_handler.subscription_handler.ressources_consommation = res_subscriptions
        #
        # messages_thread.ajouter_consumer(res_subscriptions)

        # if res_evenements.rk:
        #     messages_thread.ajouter_consumer(res_evenements)

