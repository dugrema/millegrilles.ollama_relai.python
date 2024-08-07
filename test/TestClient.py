import asyncio
import logging
import json

from millegrilles_messages.messages import Constantes
from millegrilles_messages.messages.MessagesThread import MessagesThread
from millegrilles_messages.messages.MessagesModule import RessourcesConsommation

logger = logging.getLogger(__name__)

LOGGING_FORMAT = '%(asctime)s %(threadName)s %(levelname)s: %(message)s'

RELAI_DOMAINE = 'ollama_relai'


async def main():
    logger.info("Debut main()")
    stop_event = asyncio.Event()

    # Preparer resources consumer
    reply_res = RessourcesConsommation(callback_reply_q)
    messages_thread = MessagesThread(stop_event)
    messages_thread.set_env_configuration(dict())
    messages_thread.set_reply_ressources(reply_res)

    # Demarrer traitement messages
    await messages_thread.start_async()

    tasks = [
        asyncio.create_task(messages_thread.run_async()),
        asyncio.create_task(run_tests(messages_thread, stop_event)),
    ]

    # Execution de la loop avec toutes les tasks
    await asyncio.tasks.wait(tasks, return_when=asyncio.tasks.FIRST_COMPLETED)


async def run_tests(messages_thread, stop_event):
    # Demarrer test (attendre connexion prete)
    await messages_thread.attendre_pret(max_delai=3)

    logger.info("emettre commandes")

    await run_generate(messages_thread)

    stop_event.set()

    logger.info("Fin main()")


async def run_generate(messages_thread):
    commande = {
        'model': 'llama3.1',
        'prompt': 'Why is the sky blue?',
        'stream': False,
    }
    producer = messages_thread.get_producer()
    reponse = await producer.executer_commande(commande, RELAI_DOMAINE, action='generate', exchange=Constantes.SECURITE_PRIVE)
    contenu = json.dumps(reponse.parsed, indent=2)
    logger.info("Reponse recue : %s", contenu)


async def callback_reply_q(message, messages_module):
    logger.info("Message recu : %s" % message)
    # wait_event.wait(0.7)


if __name__ == '__main__':
    logging.basicConfig(format=LOGGING_FORMAT, level=logging.WARN)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger('millegrilles_messages').setLevel(logging.DEBUG)
    logging.getLogger('millegrilles_ollama_relai').setLevel(logging.DEBUG)

    asyncio.run(main())
