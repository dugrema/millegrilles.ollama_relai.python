import asyncio
import json
import logging

from asyncio import TaskGroup
from typing import Callable, Awaitable, Optional, Any

from millegrilles_messages.bus.BusContext import ForceTerminateExecution
from millegrilles_messages.messages import Constantes
from millegrilles_messages.structs.Filehost import Filehost
from millegrilles_messages.Filehost import FilehostConnection
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.OllamaInstanceManager import OllamaInstanceManager
from millegrilles_ollama_relai.Structs import OllamaRelaiConfigurationFile, OllamaRelaiConfigurationProperties, \
    OllamaRelaiConfigurationPropertiesValue


class OllamaManager:

    def __init__(self, context: OllamaContext, ollama_instances: OllamaInstanceManager, attachment_handler: FilehostConnection):
        self.__logger = logging.getLogger(__name__+'.'+self.__class__.__name__)
        self.__context = context
        self.__ollama_instances = ollama_instances
        self.__attachment_handler = attachment_handler

        self.__filehost_listeners: list[Callable[[Optional[Filehost]], Awaitable[None]]] = list()

        self.__load_ai_configuration_event = asyncio.Event()
        self.__load_filehost_event = asyncio.Event()
        self.__ollama_available = False

        self.__current_configuration: Optional[OllamaRelaiConfigurationFile] = None     # Configuration as received
        self.__property_dict: Optional[dict] = None                                     # Parsed properties
        self.__tasks_dict: Optional[dict] = None                                        # Individual task properties

    @property
    def context(self):
        return self.__context

    async def setup(self):
        pass

    async def __stop_thread(self):
        await self.__context.wait()
        # Free threads
        self.__load_ai_configuration_event.set()
        self.__load_filehost_event.set()
        await asyncio.sleep(0.5)

    async def run(self):
        self.__logger.debug("OllamaManager Starting")
        try:
            async with TaskGroup() as group:
                group.create_task(self.__reload_filehost_thread())
                group.create_task(self.__reload_ai_configuration_thread())
                group.create_task(self.__ollama_watchdog_thread())
                group.create_task(self.__stop_thread())
        except* (asyncio.CancelledError, ForceTerminateExecution):
            if self.__context.stopping is False:
                self.__logger.warning("Ollama manager cancelled / forced to terminate without setting context to stop")
                self.__context.stop()
            else:
                self.__logger.info("OllamaManager tasks cancelled, stopping")
        self.__logger.debug("OllamaManager Done")

    def add_filehost_listener(self, listener: Callable[[Optional[Filehost]], Awaitable[None]]):
        self.__filehost_listeners.append(listener)

    async def __reload_filehost_thread(self):
        while self.__context.stopping is False:
            self.__load_filehost_event.clear()
            try:
                await self.reload_filehost_configuration()
            except asyncio.TimeoutError as e:
                self.__logger.error("Error loading filehost configuration: %s" % e)
                await self.__context.wait(15)
                continue  # Loop immediately

            try:
                await asyncio.wait_for(self.__load_filehost_event.wait(), 900)
            except asyncio.TimeoutError:
                pass  # Loop

        self.__logger.info("__reload_filehost_thread Stopping")

    async def reload_filehost_configuration(self):
        await self.__context.reload_filehost_configuration()

        for l in self.__filehost_listeners:
            await l(self.__context.filehost)

    async def trigger_reload_ai_configuration(self):
        self.__logger.info("Reloading AI Configuration on event")
        self.__load_ai_configuration_event.set()

    async def __reload_ai_configuration_thread(self):
        while self.__context.stopping is False:
            try:
                self.__load_ai_configuration_event.clear()
                await self.__reload_ai_configuration()
            except asyncio.TimeoutError:
                self.__logger.exception("Error loading ai configuration")
                await self.__context.wait(20)

            try:
                await asyncio.wait_for(self.__load_ai_configuration_event.wait(), 300)
            except asyncio.TimeoutError:
                pass  # Loop

        self.__logger.info("__reload_ai_configuration_thread Stopping")

    async def __reload_ai_configuration(self):
        producer = await self.context.get_producer()
        response = await producer.request(
            {"filename": "ollama_relai"},
            "CoreTopologie",
            "requestConfigurationGetProperties",
            exchange=Constantes.SECURITE_PUBLIC
        )
        parsed = response.parsed

        if parsed.get("ok") is False:
            if parsed.get("code") == 404:
                self.__logger.warning(f"Configuration file ollama_relai not created yet in CoreTopologie, ollama_relai will not do anything")
                return
            else:
                raise Exception(f"Error getting configuration from CoreTopologie ({parsed.get("code")}): {parsed.get("err")}")

        del parsed['__original']
        self.__logger.debug(f"Configuration received:\n{json.dumps(parsed, indent=2)}")
        await self.__process_configuration_changes(parsed)

        # For initial configuration load
        self.__context.ai_configuration_loaded.set()

    async def __ollama_watchdog_thread(self):
        """ Regularly checks status of ollama connection. """
        while self.__context.stopping is False:
            try:
                await asyncio.wait_for(self.__context.ai_configuration_loaded.wait(), 2)
            except asyncio.TimeoutError:
                continue  # Retry

            try:
                await self.__context.wait(10)
            except ForceTerminateExecution:
                pass

        self.__logger.info("__ollama_watchdog_thread Stopping")

    async def __process_configuration_changes(self, configuration: OllamaRelaiConfigurationFile):
        if self.__current_configuration is not None:
            # TODO - process changes
            self.__logger.warning("TODO - handle configuration changes, IGNORING for now")
            return

        self.__current_configuration = configuration

        property_dict, tasks_dict = parse_configuration(configuration)

        self.__logger.debug(f"Configuration:\n{json.dumps(property_dict, indent=2)}")
        self.__logger.debug(f"Tasks:\n{json.dumps(tasks_dict, indent=2)}")

        try:
            if get_property_int(property_dict, 'active') != 1:
                self.__logger.info("General ollama_relai active flag not set, not starting tasks")
                return
        except KeyError:
            return

        self.__property_dict = property_dict
        self.__tasks_dict = tasks_dict

        await self.initialize_tasks()

    async def initialize_tasks(self):
        tasks_dict = self.__tasks_dict
        if tasks_dict is None:
            raise Exception("No task configuration to process")


def get_property_text(properties: dict, key: str) -> Optional[str]:
    try:
        return properties[key]['text']
    except KeyError:
        return None

def get_property_int(properties: dict, key: str) -> Optional[int]:
    try:
        return properties[key]['inumber']
    except KeyError:
        return None

def get_property_float(properties: dict, key: str) -> Optional[float]:
    try:
        return properties[key]['fnumber']
    except KeyError:
        return None


def parse_configuration(configuration: OllamaRelaiConfigurationFile) -> tuple[dict[Any, Any], dict[Any, Any]]:
    property_dict = dict()
    task_dict = dict()
    for item in configuration['list']:
        key = item['key']
        if key.startswith("task."):
            task_key = key.split('.')
            task_name = task_key[1]
            try:
                task_values = task_dict[task_name]
            except KeyError:
                task_values = dict()
                task_dict[task_name] = task_values
            if task_key[2] == 'param':
                try:
                    task_params = task_values['params']
                except KeyError:
                    task_params = dict()
                    task_values['params'] = task_params
                task_params['.'.join(task_key[3:])] = item['value']
            else:
                task_values['.'.join(task_key[2:])] = item['value']
        else:
            property_dict[item['key']] = item['value']
    return property_dict, task_dict
