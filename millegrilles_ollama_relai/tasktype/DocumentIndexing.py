import asyncio
import datetime
import logging
from typing import Optional

from cryptography.x509 import ExtensionNotFound

from millegrilles_messages.bus.PikaChannel import MilleGrillesPikaChannel
from millegrilles_messages.bus.PikaQueue import MilleGrillesPikaQueueConsumer, RoutingKey
from millegrilles_messages.messages import Constantes
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.Structs import get_property_int


class Processor:

    def __init__(self, context: OllamaContext, task_name: str, params: dict):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context: OllamaContext = context
        self.__task_name: str = task_name
        self.__params: dict = params
        self.__channels: Optional[list[MilleGrillesPikaChannel]] = None
        self.__stop_event = asyncio.Event()
        self.__fetch_event = asyncio.Event()

    async def setup(self):
        self.__logger.info(f"Setting up DocumentIndexing task {self.__task_name} with params {self.__params}...")
        await self.__set_up_mq()

    async def __stop_thread(self):
        await self.__context.wait()
        self.__stop_event.set()
        # Release all threads
        self.__fetch_event.set()

    async def run(self):
        # Start all tasks
        self.__logger.debug(f"Running task processor {self.__task_name}")
        asyncio.create_task(self.__stop_thread())   # Sets the stop event

        # Work threads
        asyncio.create_task(self.__job_fetch_thread())

        while not self.__stop_event.is_set():
            self.__logger.debug(f"Still running task {self.__task_name}")
            try:
                self.__fetch_event.set()    # Fetches new jobs
                await asyncio.wait_for(self.__stop_event.wait(), 30)
            except asyncio.TimeoutError:
                pass

        # Stop processing
        if self.__channels:
            for channel in self.__channels:
                await channel.stop_consuming()
                await self.__context.bus_connector.remove_channel(channel)
            self.__channels = None

    async def stop(self):
        self.__logger.debug(f"Stopping task processor {self.__task_name}")
        self.__stop_event.set()

    async def tick(self):
        """
        Gets called whenever this ollama_relai instance wins a tick from ceduleur (so once a minute for all ollama_relai instances)
        :return:
        """
        pass

    async def __job_fetch_thread(self):
        while not self.__stop_event.is_set():
            self.__logger.debug("Fetching jobs")
            await self.__fetch_event.wait()

            # Potential flood of new fuuids, wait a few seconds before starting to process
            await self.__context.wait(5)
            if self.__stop_event.is_set():
                return  # Stopping
            self.__fetch_event.clear()  # Reset flag

            # TODO - do work

    async def __set_up_mq(self):
        # Initialize queue that listens to external events
        channels = list()
        worker_count = get_property_int(self.__params, 'workers') or 1
        self.__logger.debug(f"DocumentIndexing task {self.__task_name} worker count: {worker_count}")

        # Set-up the new fuuid listener, triggers job queries
        q_instance = MilleGrillesPikaQueueConsumer(
            self.__context,
            self.__on_newfuuid_event,
            f'ollama_relai/task_{self.__task_name}_triggers',
            auto_delete=True,
            arguments={'x-message-ttl': 5_000}
        )
        q_instance.add_routing_key(RoutingKey(
            Constantes.SECURITE_PUBLIC, 'evenement.filecontroler.filehostNewFuuid'))
        q_channel = MilleGrillesPikaChannel(self.__context, prefetch_count=1)
        q_channel.add_queue(q_instance)
        channels.append(q_channel)
        await self.__context.bus_connector.add_channel(q_channel)
        await q_channel.start_consuming()

        # Set-up worker queues
        for _ in range(worker_count):
            q_instance = MilleGrillesPikaQueueConsumer(
                self.__context,
                self.__on_newfuuid_event,
                f'ollama_relai/task_{self.__task_name}_workers',
                auto_delete=True,
                arguments={'x-message-ttl': 120_000}
            )

            q_instance.add_routing_key(RoutingKey(
                Constantes.SECURITE_PRIVE, f'commande.ollama_relai.task_{self.__task_name}_work'))

            q_channel = MilleGrillesPikaChannel(self.__context, prefetch_count=1)
            q_channel.add_queue(q_instance)
            channels.append(q_channel)
            await self.__context.bus_connector.add_channel(q_channel)
            await q_channel.start_consuming()
        self.__channels = channels

    async def __on_newfuuid_event(self, message: MessageWrapper):
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

        # Fuuid event messages expire after 10 seconds (or 5 seconds on queue)
        expired_timestamp = (datetime.datetime.now() - datetime.timedelta(seconds=10)).timestamp()
        if estampille < expired_timestamp:
            return None  # Ignore

        if message_type == 'evenement':
            if domain == 'filecontroler' and action == 'filehostNewFuuid' and 'filecontroler' in roles:
                self.__logger.debug(f"Task {self.__task_name} new fuuid received: {message.parsed.get('fuuid')}")

                # Trigger a new job fetch from GrosFichiers
                self.__fetch_event.set()

        return None

    async def __on_work(self, message: MessageWrapper):
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

        if message_type == 'commande':
            if action == 'work' and 'ollama_relai' in roles:
                self.__logger.debug(f"Task {self.__task_name} new work received")
                await asyncio.sleep(15)
                return None

        # self.__logger.info("__on_volatile_message Ignoring unknown action %s", message.routing_key)
        # return {'ok': False, 'code': 404, 'err': 'Unknown operation'}
        return None

