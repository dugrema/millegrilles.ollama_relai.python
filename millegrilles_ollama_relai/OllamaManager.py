import asyncio
import json
import logging

from asyncio import TaskGroup
from typing import Callable, Awaitable, Optional

from millegrilles_messages.bus.BusContext import ForceTerminateExecution
from millegrilles_messages.messages import Constantes
from millegrilles_messages.structs.Filehost import Filehost
from millegrilles_messages.Filehost import FilehostConnection
from millegrilles_ollama_relai.OllamaContext import OllamaContext
from millegrilles_ollama_relai.OllamaInstanceManager import OllamaInstanceManager


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
