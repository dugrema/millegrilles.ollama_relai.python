import asyncio
import binascii
import datetime

import httpx
import logging

from cryptography.x509 import ExtensionNotFound
from ollama import AsyncClient, ProcessResponse, ListResponse, ShowResponse
from typing import Optional, Any, Coroutine, Callable, Awaitable

from millegrilles_messages.messages import Constantes
from millegrilles_ollama_relai import Constantes as OllamaConstants
from millegrilles_messages.bus.PikaChannel import MilleGrillesPikaChannel
from millegrilles_messages.bus.PikaQueue import MilleGrillesPikaQueueConsumer, RoutingKey
from millegrilles_messages.messages.Hachage import hacher
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration
from millegrilles_ollama_relai.OllamaContext import OllamaContext

LOGGER = logging.getLogger(__name__)

class OllamaModelParams:

    def __init__(self, id: str, model: ListResponse.Model, show_response: ShowResponse):
        self.id = id
        self.model = model
        self.show_response = show_response

    @property
    def capabilities(self):
        return self.show_response.capabilities


class OllamaInstance:

    def __init__(self, context: OllamaContext, url: str, message_cb: Callable[[Any, MessageWrapper], Awaitable[Optional[dict]]],
                 status_cb: Callable[[bool], Coroutine[Any, Any, None]]):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__message_cb = message_cb
        self.url = url
        self.__status_cb = status_cb    # Callback at the end of maintenance, passes in value of ready

        self.__stop_event = asyncio.Event()
        self.__channel: Optional[MilleGrillesPikaChannel] = None
        self.__consumer: Optional[MilleGrillesPikaQueueConsumer] = None

        self.__ollama_status_response: Optional[ProcessResponse] = None
        self.__model_list_response: Optional[ListResponse] = None
        self.__ollama_model_by_id: dict[str, OllamaModelParams] = dict()

        self.__ready = False  # Ready is True when at least 1 model is online
        self.semaphore = asyncio.BoundedSemaphore(1)

    def ready(self) -> bool:
        return self.__ready

    def stop(self):
        asyncio.get_event_loop().call_soon(self.__stop_event.set)

    async def run(self):
        self.__logger.info(f"Starting ollama instance thread for {self.url}")

        # Create channel with queue consumer
        await self.__create_channel()

        while not self.__stop_event.is_set():
            await self.__maintain()
            try:
                await asyncio.wait_for(self.__stop_event.wait(), 20)
                break  # Stopping
            except asyncio.TimeoutError:
                pass

        self.__logger.info(f"Stopping ollama instance thread for {self.url}")

        # Close consumer and channel
        self.__consumer = None
        try:
            await self.__channel.close()
        except AttributeError:
            pass  # Channel not open
        finally:
            # Shutting down
            await self.__status_cb(False)

    async def __maintain(self):
        # Fetch status, models from ollama instance server
        await self.__fetch_status()
        await self.__maintain_model_keys()
        await self.__status_cb(self.__ready)

    async def __create_channel(self):
        instance_id = self.url
        instance_id = instance_id.replace('/', '_').replace(':', '_')
        channel, consumer = await create_instance_channel(self.__context, instance_id, self.__process_message)
        self.__channel = channel
        self.__consumer = consumer
        # await self.__context.bus_connector.add_channel(self.__channel)

    async def __fetch_status(self):
        self.__logger.debug(f"Checking with {self.url}")
        client = self.get_async_client(self.__context.configuration)
        try:
            # Test connection by getting currently loaded model information
            self.__ollama_status_response = await client.ps()
            self.__model_list_response = await client.list()

            current_model_ids = set(self.__ollama_model_by_id.keys())

            for model in self.__model_list_response.models:
                model_id = model_name_to_id(model.model)
                try:
                    self.__ollama_model_by_id[model_id]
                except KeyError:
                    # Add model to list
                    show_model = await client.show(model.model)
                    model_params = OllamaModelParams(model_id, model, show_model)
                    self.__ollama_model_by_id[model_id] = model_params
                else:
                    current_model_ids.remove(model_id)  # Model still used

            # Remove models that are no longer present
            for model_id in current_model_ids:
                del self.__ollama_model_by_id[model_id]

            # Status ready is True if at least 1 model is available
            self.__ready = len(self.__ollama_model_by_id) > 0

            self.__logger.debug(f"Connection OK: {self.url}")
        except (httpx.ConnectError, ConnectionError) as e:
            # Failed to connect
            if self.__logger.isEnabledFor(logging.DEBUG):
                self.__logger.exception(f"Connection error on {self.url}")
            else:
                self.__logger.info(f"Connection error on {self.url}: %s" % str(e))
            self.__ready = False  # Reset status, avoids picking this instance up

    async def __maintain_model_keys(self):
        try:
            current_rks = set(self.__consumer.routing_keys)
        except AttributeError:
            return  # Consumer not ready yet

        current_model_ids = set()
        for rk in current_rks:
            model_id = rk.routing_key.split('.')[2]
            current_model_ids.add(model_id)

        model_ids = set(self.__ollama_model_by_id.keys())
        new_model_ids = model_ids.difference(current_model_ids)
        ids_to_remove = current_model_ids.difference(model_ids)

        # Remove routing keys that are no longer handled
        for model_id in ids_to_remove:
            await self.__consumer.remove_unbind_routing_key(RoutingKey(
                Constantes.SECURITE_PROTEGE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.{model_id}.process'))

        # Add new model routing keys
        for model_id in new_model_ids:
            await self.__consumer.add_bind_routing_key(RoutingKey(
                Constantes.SECURITE_PROTEGE, f'commande.{OllamaConstants.DOMAIN_OLLAMA_RELAI}.{model_id}.process'))

    def get_client_options(self, configuration: OllamaConfiguration) -> dict:
        connection_url = self.url
        if connection_url.lower().startswith('https://'):
            # Use a millegrille certificate authentication
            cert = (configuration.cert_path, configuration.key_path)
            params = {'host':connection_url, 'verify':configuration.ca_path, 'cert':cert}
        else:
            params = {'host':connection_url}
        return params

    def get_async_client(self, configuration: OllamaConfiguration, timeout=None) -> AsyncClient:
        options = self.get_client_options(configuration)
        return AsyncClient(timeout=timeout, **options)

    @property
    def models(self) -> list[OllamaModelParams]:
        return [m for m in self.__ollama_model_by_id.values()]

    def get_model(self, model_name: str):
        model_id = model_name_to_id(model_name)
        return self.__ollama_model_by_id[model_id]

    def has_model_id(self, model_id: str) -> bool:
        return self.__ollama_model_by_id.get(model_id) is not None

    async def __process_message(self, message: MessageWrapper):
        return await self.__message_cb(self, message)


class OllamaInstanceManager:

    def __init__(self, context: OllamaContext):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context

        self.__status_event = asyncio.Event()
        self.__instances: list[OllamaInstance] = list()
        self.__ollama_ready: bool = False
        # self.ollama_models: Optional[list[Any]] = None
        # self.ollama_model_params: dict[str, dict] = dict()

        self.__group: Optional[asyncio.TaskGroup] = None
        self.__message_cb: Optional[Callable[[str, MessageWrapper], Awaitable[Optional[dict]]]] = None

        # Cache of all queries that have started processing for which the message is not yet expired
        # Used to deduplicate query processing across multiple instances.
        self.__query_dedupe_memory: dict[str, dict] = dict()

    @property
    def ready(self) -> bool:
        return self.__ollama_ready

    @property
    def ollama_instances(self) -> list[OllamaInstance]:
        return self.__instances

    def set_message_cb(self, cb: Callable[[OllamaInstance, MessageWrapper], Awaitable[Optional[dict]]]):
        self.__message_cb = cb

    async def __stop_thread(self):
        await self.__context.wait()

        # Free threads, resources
        self.__status_event.set()

        # Stop all instances
        for instance in self.__instances:
            instance.stop()

    async def run(self):

        # Check that wiring is complete
        if self.__message_cb is None:
            raise Exception('__message_cb has not been wired')

        async with asyncio.TaskGroup() as group:
            self.__group = group
            group.create_task(self.__status_thread())
            group.create_task(self.__query_cache_maintenance_thread())
            group.create_task(self.__stop_thread())

        # Cleanup
        self.__group = None

    def update_instance_list(self, urls: list[str]):
        url_set = set(urls)

        to_remove = set()
        for instance in self.__instances:
            try:
                url_set.remove(instance.url)
                self.__logger.debug(f"URL {instance.url} kept")
            except KeyError:
                # This instance has been removed
                instance.stop()
                to_remove.add(instance.url)
                self.__logger.debug(f"URL {instance.url} removed")

        # Remove instances that are no longer required
        updated_list = [i for i in self.__instances if i.url not in to_remove]

        # Add missing instances
        for url in url_set:
            instance = OllamaInstance(self.__context, url, self.__message_cb, self.__model_update_cb)
            updated_list.append(instance)
            self.__group.create_task(instance.run())
            self.__logger.debug(f"URL {url} added")

        self.__instances = updated_list

    async def __model_update_cb(self, ready: bool):
        """
        Called during maintenance of a model
        :param ready:
        :return:
        """
        self.__status_event.set()

    async def __status_thread(self):
        while self.__context.stopping is False:
            self.__status_event.clear()

            ready = False

            # Make a list of all available models
            model_by_name_dict: dict[str, OllamaModelParams] = dict()
            for instance in self.__instances:
                # Toggle value for ready if at least one instance is ready
                if instance.ready():
                    ready = True

                models = instance.models
                for model in models:
                    model_by_name_dict[model.model.model] = model

            if ready != self.__ollama_ready:
                self.__ollama_ready = ready

                # Emit status event
                self.__logger.info(f"ollama Status now: {ready}")
                producer = await self.__context.get_producer()
                status_event = {'event_type': 'availability', 'available': ready}
                await producer.event(status_event, 'ollama_relai', 'status', exchange=Constantes.SECURITE_PRIVE)

            self.__model_by_name_dict = model_by_name_dict

            # Wait until next model/status event
            await self.__status_event.wait()

    def get_models(self):
        models = list()
        for model in self.__model_by_name_dict.values():
            models.append({
                'name': model.model.model,
                'capabilities': model.capabilities,
            })
        return models

    async def __query_cache_maintenance_thread(self):
        while self.__context.stopping is False:

            expired = datetime.datetime.now() - datetime.timedelta(seconds=900)
            to_remove = list()

            for (k, v) in self.__query_dedupe_memory.items():
                if v['date'] < expired:
                    to_remove.append(k)

            for q_id in to_remove:
                del self.__query_dedupe_memory[q_id]

            await self.__context.wait(180)

    def claim_query(self, query_id: str):
        try:
            _info = self.__query_dedupe_memory[query_id]  # KeyError if not present
            raise Exception('Already processing')
        except KeyError:
            # Ok, assign to process
            self.__query_dedupe_memory[query_id] = {'date': datetime.datetime.now()}

    def pick_instance_for_model(self, model: str) -> Optional[OllamaInstance]:
        model_id = model_name_to_id(model)

        instances = [i for i in self.__instances if i.has_model_id(model_id)]

        for i in instances:
            if not i.semaphore.locked():
                return i

        # All instances locked, return one at random
        try:
            return instances[0]
        except IndexError:
            return None

async def create_instance_channel(context: OllamaContext,
                                  instance_id: str,
                                  on_message: Callable[[MessageWrapper], Coroutine[Any, Any, None]]) -> (MilleGrillesPikaChannel, MilleGrillesPikaQueueConsumer):

    nom_ou = context.signing_key.enveloppe.subject_organizational_unit_name
    if nom_ou is None:
        raise Exception('Invalid certificate - no Organizational Unit (OU) name')
    queue_name = f'{nom_ou}/processing/{instance_id}'

    q_channel = MilleGrillesPikaChannel(context, prefetch_count=1)
    q_instance = MilleGrillesPikaQueueConsumer(context, on_message, queue_name, arguments={'x-message-ttl': 900_000})

    await context.bus_connector.add_channel(q_channel)

    try:
        await asyncio.wait_for(q_channel.ready.wait(), 5)
    except asyncio.TimeoutError:
        LOGGER.warning("Timeout waiting for q_channel ready, starting consumer")
        await q_channel.start_consuming()

    await q_channel.add_queue_consume(q_instance)

    return q_channel, q_instance


def model_name_to_id(name: str) -> str:
    """
    :param name: Model name
    :return: A 16 char model id
    """
    return hacher(name.lower(), hashing_code='blake2s-256')[-16:]
