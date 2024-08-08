import asyncio
import binascii
import logging
import json

from typing import Optional

from pypdf import PdfReader

from millegrilles_messages.messages import Constantes
from millegrilles_messages.messages.MessagesThread import MessagesThread
from millegrilles_messages.messages.MessagesModule import RessourcesConsommation, MessageWrapper

logger = logging.getLogger(__name__)

LOGGING_FORMAT = '%(asctime)s %(threadName)s %(levelname)s: %(message)s'

RELAI_DOMAINE = 'ollama_relai'

# Global
CHAT_INSTANCE = None


async def main():
    logger.info("Debut main()")
    stop_event = asyncio.Event()

    # Preparer resources consumer
    reply_res = RessourcesConsommation(callback_reply_q)
    reply_res.ajouter_rk(Constantes.SECURITE_PRIVE, "evenement.ollama_relai.*.attente")
    reply_res.ajouter_rk(Constantes.SECURITE_PRIVE, "evenement.ollama_relai.*.debutTraitement")
    reply_res.ajouter_rk(Constantes.SECURITE_PRIVE, "evenement.ollama_relai.*.encours")
    reply_res.ajouter_rk(Constantes.SECURITE_PRIVE, "evenement.ollama_relai.*.resultat")
    reply_res.ajouter_rk(Constantes.SECURITE_PRIVE, "evenement.ollama_relai.*.annule")

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

    # await run_generate(messages_thread)
    # await run_image(messages_thread)
    # await run_pdf(messages_thread)
    await run_chat(messages_thread)

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


async def run_image(messages_thread):

    with open('/home/mathieu/tas/work/001.JPG', 'rb') as fichier:
        image = binascii.b2a_base64(fichier.read()).decode('utf-8')

    commande = {
        'model': 'llava-llama3',
        'prompt': 'Please describe this image.',
        'images': [image],
        'stream': False,
    }

    producer = messages_thread.get_producer()
    reponse = await producer.executer_commande(commande, RELAI_DOMAINE, action='generate', exchange=Constantes.SECURITE_PRIVE)
    contenu = json.dumps(reponse.parsed, indent=2)
    logger.info("Reponse recue : %s", contenu)


async def run_pdf(messages_thread):

    pdf_file = PdfReader('/home/mathieu/tas/work/WorldMarkets.pdf')

    file_content = ''
    for page in pdf_file.pages:
        file_content += page.extract_text() + '\n'

    print("File content\n%s" % file_content)

    commande = {
        'model': 'llama3.1',
        'prompt': 'Please summarise the following document.\n\n%s' % file_content,
        'stream': False,
    }

    producer = messages_thread.get_producer()
    reponse = await producer.executer_commande(commande, RELAI_DOMAINE, action='generate', exchange=Constantes.SECURITE_PRIVE)
    contenu = json.dumps(reponse.parsed, indent=2)
    logger.info("Reponse recue : %s", contenu)


async def run_chat(messages_thread):
    global CHAT_INSTANCE
    CHAT_INSTANCE = Chat()
    await CHAT_INSTANCE.run_chat(messages_thread)


class Chat:

    def __init__(self):
        self.correlation_id: Optional[str] = None
        self.response_q: asyncio.Queue[MessageWrapper] = asyncio.Queue(maxsize=1)

    async def run_chat(self, messages_thread):

        source_messages = [
            {'role': 'user', 'content': 'Why is the sky blue?'},
            {'role': 'user', 'content': 'Could you say more about Rayleigh scattering?'},
        ]

        messages = list()
        for i in range(0, len(source_messages)):
            messages.append(source_messages[i])
            commande = {'model': 'llama3.1', 'messages': messages, 'stream': False}

            producer = messages_thread.get_producer()

            # Recuperer le message_id pour faire la correlation
            commande, self.correlation_id = await producer.signer(commande, Constantes.KIND_COMMANDE,
                                                                  RELAI_DOMAINE, action='chat')
            reponse = await producer.executer_commande(commande, RELAI_DOMAINE, action='chat',
                                                       exchange=Constantes.SECURITE_PRIVE, noformat=True)

            contenu = json.dumps(reponse.parsed, indent=2)
            logger.info("Reponse recue : %s", contenu)

            reponse = await asyncio.wait_for(self.response_q.get(), 600)
            print("Reponse: %s" % reponse.parsed)
            message_assistant = reponse.parsed['message']
            messages.append(message_assistant)


async def callback_reply_q(message: MessageWrapper, messages_module):
    global CHAT_INSTANCE
    logger.info("Message recu : %s" % message)

    if message.kind == Constantes.KIND_REPONSE_CHIFFREE:
        # Reponse
        rk_split = message.routing_key.split('.')
        action = rk_split.pop()
        partition = rk_split.pop()
    else:
        partition = message.routage['partition']
        action = message.routage['action']

    if CHAT_INSTANCE:
        if action == 'resultat' and partition == CHAT_INSTANCE.correlation_id:
            await CHAT_INSTANCE.response_q.put(message)


if __name__ == '__main__':
    logging.basicConfig(format=LOGGING_FORMAT, level=logging.WARN)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logging.getLogger('millegrilles_messages').setLevel(logging.DEBUG)
    logging.getLogger('millegrilles_ollama_relai').setLevel(logging.DEBUG)

    asyncio.run(main())
