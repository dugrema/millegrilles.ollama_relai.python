import asyncio
import datetime
import logging
import json
from typing import Optional

from cryptography.x509 import ExtensionNotFound
from multibase import multibase

from millegrilles_messages.bus.PikaChannel import MilleGrillesPikaChannel
from millegrilles_messages.bus.PikaQueue import MilleGrillesPikaQueueConsumer, RoutingKey
from millegrilles_messages.chiffrage.DechiffrageUtils import dechiffrer_document, dechiffrer_bytes_secrete
from millegrilles_messages.chiffrage.Mgs4 import chiffrer_document_cles
from millegrilles_messages.messages import Constantes
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.DocumentIndexHandler import FileInformation
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.Structs import get_property_int

CONST_ACTION_SUMMARY = 'leaseForSummary'
CONST_JOB_SUMMARY_TEXT = 'summaryText'
CONST_JOB_SUMMARY_IMAGE = 'summaryImage'

class Processor:

    def __init__(self, context: OllamaContext, task_name: str, params: dict):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context: OllamaContext = context
        self.__task_name: str = task_name
        self.__params: dict = params
        self.__channels: Optional[list[MilleGrillesPikaChannel]] = None
        self.__stop_event = asyncio.Event()
        self.__fetch_event = asyncio.Event()
        self.__fetch_batch_size = 20

    @property
    def routing_action_work(self):
        return f'task_{self.__task_name}_work'

    async def setup(self):
        self.__logger.info(f"Setting up DocumentIndexing task {self.__task_name} with params {self.__params}...")
        await self.__set_up_mq()

        self.__fetch_batch_size = get_property_int(self.__params, 'batchsize') or 20

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
            # Potential flood of new fuuids, wait a few seconds before starting to process
            await self.__context.wait(5)
            if self.__stop_event.is_set():
                return  # Stopping
            self.__fetch_event.clear()  # Reset flag

            self.__logger.debug(f"Task {self.__task_name} fetching new jobs")
            try:
                await self.__query_batch()
            except:
                self.__logger.exception("Error fetching batch of documents")

            await self.__fetch_event.wait()

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
                self.__on_work,
                f'ollama_relai/task_{self.__task_name}_workers',
                auto_delete=True,
                arguments={'x-message-ttl': 120_000}
            )

            q_instance.add_routing_key(RoutingKey(
                Constantes.SECURITE_PRIVE,
                f'commande.ollama_relai.{self.routing_action_work}'
            ))

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
            if action == self.routing_action_work and 'ollama_relai' in roles:
                self.__logger.debug(f"Task {self.__task_name} new work received")
                try:
                    await self.__process_work_item(message.parsed)
                except:
                    self.__logger.exception("Error processing work item")
                return None
            else:
                self.__logger.debug(f"Task {self.__task_name} unhandled action received: {action}")
        else:
            self.__logger.debug(f"Task {self.__task_name} unhandled message type received: {message_type}")

        # self.__logger.info("__on_volatile_message Ignoring unknown action %s", message.routing_key)
        # return {'ok': False, 'code': 404, 'err': 'Unknown operation'}
        return None

    async def __query_batch(self):
        producer = await self.__context.get_producer()

        try:
            filehost_id = self.__context.filehost.filehost_id
        except AttributeError:
            # Filehost not loaded yet, wait and retry
            await self.__context.wait(5)
            filehost_id = self.__context.filehost.filehost_id

        if not filehost_id:
            self.__logger.warning(f"No filehost id provided, skipping document indexing")
            return

        # Get the batch
        command = {"batch_size": self.__fetch_batch_size, "filehost_id": filehost_id}
        try:
            response = await producer.command(command, Constantes.DOMAINE_GROS_FICHIERS, CONST_ACTION_SUMMARY,
                                              Constantes.SECURITE_PROTEGE, timeout=60)
        except asyncio.TimeoutError:
            self.__logger.warning(f"Timeout on {CONST_ACTION_SUMMARY}, will retry")
            return

        parsed = response.parsed
        if parsed['ok'] is not True:
            self.__logger.warning("Error retrieving batch of files: %s" % parsed)
        elif parsed.get('code') == 1:
            self.__logger.debug(f"No more files to process for {CONST_ACTION_SUMMARY} on filehost_id {filehost_id}")
            return

        # Parse batch file and insert on queue per item
        leases = parsed['leases']
        secret_keys: list = parsed['secret_keys']

        self.__logger.debug(f"Received batch of {len(leases)} files for {CONST_ACTION_SUMMARY} from filehost_id:{filehost_id}")

        for lease in leases:
            metadata = lease.get('metadata')
            version = lease.get('version')
            cuuids = lease.get('cuuids')
            fuuid: Optional[str] = None

            cle_id = None
            mimetype = lease.get('mimetype')
            if version:
                fuuid = version['fuuid']
                cle_id = version.get('cle_id')
                mimetype = mimetype or version.get('mimetype')

            if metadata and not cle_id:
                cle_id = cle_id or metadata.get('cle_id') or metadata.get('ref_hachage_bytes')
            cle_id = cle_id or fuuid

            try:
                key = [k for k in secret_keys if k['cle_id'] == cle_id].pop()
            except IndexError:
                self.__logger.warning(f"Missing key for fuuid {fuuid}, canceling")
                await self.__cancel_job(fuuid)
                continue

            image_file = None
            media = lease.get('media')
            if mimetype == 'application/pdf' or mimetype.startswith('text/'):
                job_type = CONST_JOB_SUMMARY_TEXT
            elif mimetype.startswith('image/'):
                job_type = CONST_JOB_SUMMARY_IMAGE
            else:
                self.__logger.info(f"Unsupported mimetype {mimetype} for fuuid {fuuid}, canceling")
                await self.__cancel_job(fuuid)
                continue

            try:
                # Check to find a webp fuuid, will be smaller and easier to digest
                images = media['images']
                webp_img = [m for m in images.keys() if m.startswith('image/webp')].pop()
                image_file = images[webp_img]
                image_file['fuuid'] = image_file['hachage']
            except (AttributeError, IndexError, TypeError):
                pass

            self.__logger.debug("Lease:\n%s", lease)
            secret_key: bytes = multibase.decode('m'+key['cle_secrete_base64'])
            decrypted_metadata = json.loads(dechiffrer_bytes_secrete(secret_key, metadata))

            info: FileInformation = {
                'job_type': job_type,
                'lease_action': CONST_ACTION_SUMMARY,
                'tuuid': lease.get('tuuid'),
                'fuuid': lease.get('fuuid') or fuuid,
                'user_id': lease['user_id'],
                'language': 'en_US',
                'domain': Constantes.DOMAINE_GROS_FICHIERS,
                'cuuids': cuuids,
                'metadata': decrypted_metadata,
                'mimetype': lease.get('mimetype'),
                'version': lease.get('version'),
                'key': key,
                'tmp_file': None,
                'image_tmp_file': None,
                'media': lease.get('media'),
                'image_file': image_file,
            }

            # Encrypt the contents to put in a queue
            # TODO - figure out a way to encrypt for all ollama_relai instances
            certs = [self.__context.signing_key.enveloppe]
            encrypted_job = chiffrer_document_cles(certs, info)

            # Put encrypted job on work queue
            try:
                await producer.command(encrypted_job, 'ollama_relai', self.routing_action_work, Constantes.SECURITE_PRIVE, nowait=True)
            except asyncio.TimeoutError:
                self.__logger.warning(f"Timeout on submitting {job_type}, will retry")
                return

        pass

    async def __process_work_item(self, job: dict):
        self.__logger.debug("Processing work item\n%s", job)
        signing_key = self.__context.signing_key
        fingerprint = signing_key.fingerprint
        try:
            encrypted_key = job['cles'][fingerprint]
        except KeyError:
            self.__logger.error("No keys available to decrypt job, skipping")
            return

        # Decrypt key
        decrypted_job = dechiffrer_document(signing_key, encrypted_key, job)
        self.__logger.debug("Decrypted job\n%s" % decrypted_job)

    async def __cancel_job(self, fuuid):
        self.__logger.debug(f"Canceling job on fuuid {fuuid}")
        # TODO
        pass
