import asyncio
import datetime

import httpx
import logging

import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from ollama import AsyncClient as OllamaAsyncClient, ListResponse, ResponseError as OllamaResponseError
from openai import AsyncClient as OpenaiAsyncClient, OpenAIError
from typing import Optional, Any, Coroutine, Callable, Awaitable

from millegrilles_messages.messages import Constantes
from millegrilles_ollama_relai import Constantes as OllamaConstants
from millegrilles_messages.bus.PikaChannel import MilleGrillesPikaChannel
from millegrilles_messages.bus.PikaQueue import MilleGrillesPikaQueueConsumer, RoutingKey
from millegrilles_messages.messages.MessagesModule import MessageWrapper
from millegrilles_ollama_relai.InstancesDao import InstanceDao, OllamaModelParams, OpenAiInstanceDao, OllamaInstanceDao, \
    ClientDaoError
from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.Util import model_name_to_id

LOGGER = logging.getLogger(__name__)

CONST_OLLAMA_LEASE_DURATION = 20
CONST_OLLAMA_CHAT_LEASE_DURATION = 900  # Message duration in queue


class OllamaInstance:

    def __init__(self, context: OllamaContext, url: str, update_lease_cb: Callable[[str], Awaitable[bool]],
                 message_cb: Callable[[Any, MessageWrapper], Awaitable[Optional[dict]]],
                 status_cb: Callable[[bool], Coroutine[Any, Any, None]]):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context
        self.__message_cb = message_cb
        self.__update_lease_cb = update_lease_cb
        self.url = url
        self.__status_cb = status_cb    # Callback at the end of maintenance, passes in value of ready

        self.__stop_event = asyncio.Event()
        self.__channel: Optional[MilleGrillesPikaChannel] = None
        self.__consumer: Optional[MilleGrillesPikaQueueConsumer] = None

        self.__ollama_status_active = False
        self.__model_list_response: Optional[ListResponse] = None
        self.__ollama_model_by_id: dict[str, OllamaModelParams] = dict()

        self.__ready = False  # Ready is True when at least 1 model is online
        self.semaphore = asyncio.BoundedSemaphore(1)
        self.__connection_dao: Optional[InstanceDao] = None

    def ready(self) -> bool:
        return self.__ready

    def stop(self):
        asyncio.get_event_loop().call_soon(self.__stop_event.set)

    async def __detect_instance_type(self):
        url = self.url

        while self.__context.stopping is False:
            try:
                client = OpenaiAsyncClient(api_key="DUMMY", base_url=url)
                try:
                    await client.models.list()
                    self.__logger.info(f"Connected to OpenAI at {url}")
                    # This is an openai instance
                    self.__connection_dao = OpenAiInstanceDao(self.__context.configuration, url)
                    return
                except OpenAIError:
                    pass  # Not OpenAI

                client = OllamaAsyncClient(host=url)
                try:
                    await client.ps()
                    # No error, this is an ollama instance
                    self.__logger.info(f"Connected to Ollama at {url}")
                    self.__connection_dao = OllamaInstanceDao(self.__context.configuration, url)
                    return
                except OllamaResponseError:
                    pass  # Not ollama
            except (httpx.ConnectError, ConnectionError):
                self.__logger.warning(f"Unable to connect to {url}, will retry")
            await self.__context.wait(10)

    async def run(self):
        self.__logger.info(f"Starting ollama instance thread for {self.url}")

        # Detect type of connection (Ollama, OpenAI)
        await self.__detect_instance_type()

        # Create channel with queue consumer
        await self.__create_channel()

        async with asyncio.TaskGroup() as group:
            group.create_task(self.__refresh_thread())
            group.create_task(self.__lease_thread())

        self.__logger.info(f"Stopping instance thread for {self.url}")

        # Close consumer and channel
        self.__consumer = None
        try:
            await self.__channel.close()
        except AttributeError:
            pass  # Channel not open
        finally:
            # Shutting down
            await self.__status_cb(False)

    async def __lease_thread(self):
        while not self.__stop_event.is_set():
            await self.__update_lease_cb(self.url)
            try:
                await asyncio.wait_for(self.__stop_event.wait(), CONST_OLLAMA_LEASE_DURATION - 5)
                break  # Stopping
            except asyncio.TimeoutError:
                pass

    async def __refresh_thread(self):
        while not self.__stop_event.is_set():
            await self.__maintain()
            try:
                await asyncio.wait_for(self.__stop_event.wait(), 20)
                break  # Stopping
            except asyncio.TimeoutError:
                pass

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
        current_status = self.__ollama_status_active
        try:
            self.__ollama_status_active = await self.__connection_dao.status()
            send_update_event = current_status != self.__ollama_status_active

            model_update = await self.__connection_dao.models()
            self.__ollama_model_by_id = model_update['models']

            if len(model_update['added']) > 0 or len(model_update['removed']) > 0:
                send_update_event = True

            current_ready = self.__ready
            self.__ready = len(self.__ollama_model_by_id) > 0
            if current_ready != self.__ready:
                send_update_event = True

        except (ConnectionError, httpx.ConnectError, ClientDaoError) as e:
            # Failed to connect
            if self.__logger.isEnabledFor(logging.DEBUG):
                self.__logger.exception(f"Connection error on {self.url}")
            else:
                self.__logger.info(f"Connection error on {self.url}: %s" % str(e))

            # Reset status, avoids picking this instance up
            self.__ready = False
            self.__ollama_status_active = False
            send_update_event = current_status is not False
            self.__model_list_response = None
            self.__ollama_model_by_id.clear()

        if send_update_event:
            await self.send_model_update_event()


    # client = self.get_async_client(self.__context.configuration)
        # try:
        #     # Test connection by getting currently loaded model information
        #     self.__ollama_status_response = await client.ps()
        #     self.__model_list_response = await client.list()
        #
        #     current_model_ids = set(self.__ollama_model_by_id.keys())
        #
        #     send_update_event = False
        #     for model in self.__model_list_response.models:
        #         model_id = model_name_to_id(model.model)
        #         update = False
        #         try:
        #             existing_model = self.__ollama_model_by_id[model_id]
        #             if existing_model.model.modified_at < model.modified_at:
        #                 # Model has been updated
        #                 update = True
        #         except KeyError:
        #             update = True
        #         else:
        #             current_model_ids.remove(model_id)  # Model still used
        #
        #         if update:
        #             # Add model to list
        #             show_model = await client.show(model.model)
        #             model_params = OllamaModelParams(model_id, model, show_model)
        #             self.__ollama_model_by_id[model_id] = model_params
        #
        #             send_update_event = True  # Model added/updated
        #
        #     # Remove models that are no longer present
        #     for model_id in current_model_ids:
        #         send_update_event = True  # Model removed
        #         del self.__ollama_model_by_id[model_id]
        #
        #     # Status ready is True if at least 1 model is available
        #     self.__ready = len(self.__ollama_model_by_id) > 0
        #
        #     if send_update_event:
        #         await self.send_model_update_event()
        #
        #     self.__logger.debug(f"Connection OK: {self.url}")
        # except (ConnectionError, httpx.ConnectError, ResponseError) as e:
        #     # Failed to connect
        #     if self.__logger.isEnabledFor(logging.DEBUG):
        #         self.__logger.exception(f"Connection error on {self.url}")
        #     else:
        #         self.__logger.info(f"Connection error on {self.url}: %s" % str(e))
        #
        #     # Reset status, avoids picking this instance up
        #     self.__ready = False
        #     self.__ollama_status_response = None
        #     self.__model_list_response = None
        #     self.__ollama_model_by_id.clear()

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

    def get_async_client(self, configuration: OllamaConfiguration, timeout=None) -> OllamaAsyncClient:
        options = self.get_client_options(configuration)
        return OllamaAsyncClient(timeout=timeout, **options)

    @property
    def connection(self) -> Optional[InstanceDao]:
        return self.__connection_dao

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

    async def send_model_update_event(self):
        producer = await self.__context.get_producer()
        event = {'url': self.url}
        await producer.event(event, 'ollama_relai', 'modelsUpdated', exchange=Constantes.SECURITE_PRIVE)


class OllamaInstanceManager:

    def __init__(self, context: OllamaContext):
        self.__logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)
        self.__context = context

        self.__status_event = asyncio.Event()
        self.__ollama_instance_urls: Optional[list[str]] = None
        self.__instances: list[OllamaInstance] = list()
        self.__ollama_ready: bool = False

        self.__group: Optional[asyncio.TaskGroup] = None
        self.__message_cb: Optional[Callable[[str, MessageWrapper], Awaitable[Optional[dict]]]] = None

        # Cache of all queries that have started processing for which the message is not yet expired
        # Used to deduplicate query processing across multiple instances.
        self.__query_dedupe_memory: dict[str, dict] = dict()
        self.__redis_client: Optional[redis.Redis] = None
        self.__redis_client_semaphore = asyncio.BoundedSemaphore(1)

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

        await self.__connect_redis()

        # Check that wiring is complete
        if self.__message_cb is None:
            raise Exception('__message_cb has not been wired')

        async with asyncio.TaskGroup() as group:
            self.__group = group
            group.create_task(self.__maintain_instance_list_thread())
            group.create_task(self.__status_thread())
            group.create_task(self.__query_cache_maintenance_thread())
            group.create_task(self.__maintain_redis())
            group.create_task(self.__stop_thread())
            group.create_task(self.__cleanup_stop())

        # Cleanup. Expire locks
        self.__group = None

    async def __cleanup_stop(self):
        """
        Release all instances locked in redis
        :return:
        """
        async def shielded_stop():
            await self.__context.wait()
            for instance in self.__instances:
                if instance.ready():
                    self.__logger.info(f"Releasing {instance.url}")
                    await self.__expire_instance_lock(instance.url)
        await asyncio.shield(shielded_stop())

    async def __maintain_instance_list_thread(self):
        await self.__context.ai_configuration_loaded.wait()
        while not self.__context.stopping:
            urls = self.__ollama_instance_urls
            if urls:
                await self.update_instance_list(urls)
            await self.__context.wait(30)

    async def update_instance_list(self, urls: list[str]):
        self.__ollama_instance_urls = urls
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
        for instance_url in to_remove:
            await self.__expire_instance_lock(instance_url)

        # Add missing instances
        for url in url_set:
            # Lock the ollama instance by url - avoids running the same instance from multiple relays
            available = await self.__lease_instance(url, True)
            if available:
                instance = OllamaInstance(self.__context, url, self.__lease_instance, self.__message_cb, self.__model_update_cb)
                updated_list.append(instance)
                self.__group.create_task(instance.run())
                self.__logger.debug(f"URL {url} added")
            else:
                self.__logger.debug(f"URL {url} already locked")

        self.__instances = updated_list

    async def __expire_instance_lock(self, url: str):
        url_key = url.replace('/', '_').replace(':', '_')
        await self.__redis_client.expire(f'ollama.{url_key}', 0)

    async def __lease_instance(self, url: str, initial=False) -> bool:
        """
        :param url: Url of the ollama instance
        :param initial: True if this is for the initial lock.
        :return: True if lock acquired.
        """
        url_key = url.replace('/', '_').replace(':', '_')
        async with self.__redis_client_semaphore:
            result = await self.__redis_client.set(f'ollama.{url_key}', '', nx=initial, ex=CONST_OLLAMA_LEASE_DURATION)
        return result is True

    # async def reserve_chat_id(self, chat_id: str):
    #     """
    #     Dedupes processing of chats if multiple instances have the same models
    #     :param chat_id:
    #     :return:
    #     """
    #     key = f'ollama.chat.{chat_id}'
    #     async with self.__redis_client_semaphore:
    #         # result = await self.__redis_client.set(key, '', nx=True, ex=CONST_OLLAMA_CHAT_LEASE_DURATION)
    #         result = await self.__redis_client.setnx(key, '')
    #         await self.__redis_client.wait(1, 1)
    #         # await self.__context.wait(0.5)
    #     await self.__redis_client.expire(key, CONST_OLLAMA_CHAT_LEASE_DURATION)
    #     self.__logger.info(f"Chat_id {key} redis result: {result}")
    #     return result is True

    # async def keep_instance_lease_alive(self, url: str):
    #     """
    #     Update the instance url lock with ttl of 20 seconds
    #     :param url:
    #     :return:
    #     """
    #     url_key = url.replace('/', '_').replace(':', '_')
    #     await self.__redis_client.set(f'ollama.{url_key}', '', ex=CONST_OLLAMA_LEASE_DURATION)

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
                'num_ctx': model.context_length,
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

    async def claim_query(self, query_id: str):
        # Try to use redis first
        if self.__redis_client:
            query_key = f'ollama.query.{query_id}'
            async with self.__redis_client_semaphore:
                result = await self.__redis_client.set(query_key, '', nx=True, ex=CONST_OLLAMA_CHAT_LEASE_DURATION)
                await self.__redis_client.wait(1, 1)
            if result is not True:
                raise Exception('Already processing')
            # await self.__redis_client.expire(query_key, 900)  # Expire when all messages in MQ expire (Q TTL)
            return None

        # Fallback with local memory locks
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

    async def __maintain_redis(self):
        while self.__context.stopping is False:
            await self.__connect_redis()
            await self.__context.wait(60)

    async def __connect_redis(self):
        try:
            client = self.__redis_client
            if client is None:
                ca_pem = '\n'.join(self.__context.ca.chaine_pem())
                configuration = self.__context.configuration
                with open(configuration.redis_password_path) as file:
                    password = file.read().strip()

                client = redis.Redis(host=configuration.redis_hostname, port=configuration.redis_port,
                                     username=configuration.redis_username,
                                     password=password,
                                     ssl=True,
                                     ssl_keyfile=configuration.key_path,
                                     ssl_certfile=configuration.cert_path,
                                     ssl_ca_data=ca_pem)

                await client.select(OllamaConstants.REDIS_SESSIONS_DB)
            else:
                # Test connection
                await client.ping()

            self.__redis_client = client
            self.__logger.debug("Connection to redis OK")
        except RedisConnectionError:
            self.__redis_client = None
            self.__logger.error("Redis connection error, closing")


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
