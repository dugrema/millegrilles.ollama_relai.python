import asyncio
import logging

from millegrilles_ollama_relai.InstancesDao import OllamaInstanceDao, OpenAiInstanceDao, InstanceDao
from millegrilles_ollama_relai.OllamaConfiguration import OllamaConfiguration


LOGGER = logging.getLogger(__name__)

def setup():
    configuration = OllamaConfiguration()
    configuration.load()  # Loads logging, env params
    return configuration

def prep_ollama(configuration: OllamaConfiguration):
    ollama_dao = OllamaInstanceDao(configuration, 'http://localhost:11434')
    return ollama_dao

def prep_openai(configuration: OllamaConfiguration):
    openai_dao = OpenAiInstanceDao(configuration, 'http://localhost:8001')
    return openai_dao

async def models(dao: InstanceDao):
    models = await dao.models()
    for model in models:
        LOGGER.debug(f"Model: {model}")


async def run_test(dao: InstanceDao):
    LOGGER.debug(f"\n-----------------\nTesting {dao.__class__.__name__}")
    ready = await dao.status()
    LOGGER.debug(f"Ready: {ready}")
    if not ready:
        LOGGER.warning("OFFLINE")
        return

    LOGGER.debug(f"\n-----------------\n")


async def main():
    configuration = setup()
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    daos = [
        prep_ollama(configuration),
        prep_openai(configuration)
    ]
    for dao in daos:
        try:
            await run_test(dao)
        except NotImplementedError:
            LOGGER.exception(f'Error running {dao.__class__.__name__}')


if __name__ == '__main__':
    asyncio.run(main())
