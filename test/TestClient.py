import asyncio
import binascii
import logging
import json
import requests

from typing import Optional

from pypdf import PdfReader

from millegrilles_messages.messages import Constantes
from millegrilles_messages.messages.MessagesThread import MessagesThread
from millegrilles_messages.messages.MessagesModule import RessourcesConsommation, MessageWrapper
from millegrilles_messages.messages.EnveloppeCertificat import EnveloppeCertificat

logger = logging.getLogger(__name__)

LOGGING_FORMAT = '%(asctime)s %(threadName)s %(levelname)s: %(message)s'

RELAI_DOMAINE = 'ollama_relai'

# Global
CHAT_INSTANCE = None
EVENT_WAIT: Optional[asyncio.Event] = None


async def main():
    global EVENT_WAIT
    logger.info("Debut main()")
    stop_event = asyncio.Event()
    EVENT_WAIT = asyncio.Event()

    # Preparer resources consumer
    reply_res = RessourcesConsommation(callback_reply_q)
    reply_res.ajouter_rk(Constantes.SECURITE_PRIVE, "evenement.ollama_relai.*.evenement")

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
    global EVENT_WAIT
    enveloppes = await asyncio.to_thread(get_maitredescles_certificates)

    commande = {
        'model': 'llama3.1',
        'prompt': 'Why is the sky blue?',
        'stream': False,
    }
    producer = messages_thread.get_producer()

    commande, correlation_id = await producer.chiffrer(
        enveloppes, Constantes.KIND_COMMANDE_INTER_MILLEGRILLE, commande, domaine=RELAI_DOMAINE, action='generate')
    async for reponse in producer.executer_commande_stream(commande, RELAI_DOMAINE, action='generate',
                                                           exchange=Constantes.SECURITE_PRIVE, noformat=True):
        parsed = reponse.parsed.copy()
        del parsed['__original']
        contenu = json.dumps(parsed, indent=2)
        logger.info("Message stream recu : %s" % contenu)
        try:
            token = parsed['message']['content']
            print(token, end='')
        except:
            print('Keep alive: %s' % contenu)

    print("run_generate Done")
    EVENT_WAIT.set()
    # await EVENT_WAIT.wait()


async def run_image(messages_thread):
    enveloppes = await asyncio.to_thread(get_maitredescles_certificates)

    with open('/home/mathieu/tas/work/001.JPG', 'rb') as fichier:
        image = binascii.b2a_base64(fichier.read()).decode('utf-8')

    commande = {
        'model': 'llava-llama3',
        'prompt': 'Please describe this image.',
        'images': [image],
        'stream': False,
    }

    producer = messages_thread.get_producer()
    commande, correlation_id = await producer.chiffrer(
        enveloppes, Constantes.KIND_COMMANDE_INTER_MILLEGRILLE, commande, domaine=RELAI_DOMAINE, action='generate')
    reponse = await producer.executer_commande(commande, RELAI_DOMAINE, action='generate',
                                               exchange=Constantes.SECURITE_PRIVE, noformat=True)
    contenu = json.dumps(reponse.parsed, indent=2)
    logger.info("Confirmation recue : %s", contenu)

    EVENT_WAIT.clear()
    await EVENT_WAIT.wait()


async def run_pdf(messages_thread):
    enveloppes = await asyncio.to_thread(get_maitredescles_certificates)

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
    commande, correlation_id = await producer.chiffrer(
        enveloppes, Constantes.KIND_COMMANDE_INTER_MILLEGRILLE, commande, domaine=RELAI_DOMAINE, action='generate')
    reponse = await producer.executer_commande(commande, RELAI_DOMAINE, action='generate',
                                               exchange=Constantes.SECURITE_PRIVE, noformat=True)
    contenu = json.dumps(reponse.parsed, indent=2)
    logger.info("Confirmation recue : %s", contenu)

    EVENT_WAIT.clear()
    await EVENT_WAIT.wait()


async def run_chat(messages_thread):
    global CHAT_INSTANCE
    CHAT_INSTANCE = Chat()
    await CHAT_INSTANCE.run_chat(messages_thread)


class Chat:

    def __init__(self):
        self.correlation_id: Optional[str] = None
        self.response_q: asyncio.Queue[MessageWrapper] = asyncio.Queue(maxsize=1)

    async def run_chat(self, messages_thread):
        enveloppes = await asyncio.to_thread(get_maitredescles_certificates)

        source_messages = [
            {'role': 'user', 'content': 'Why is the sky blue?'},
            {'role': 'user', 'content': 'Could you say more about Rayleigh scattering?'},
            {'role': 'user', 'content': 'I\'d like to know more about related topics.'},
        ]

        messages = list()
        for i in range(0, len(source_messages)):

            message_user_courant = source_messages[i]
            print("User:\n%s\nReply:\n" % message_user_courant['content'])

            messages.append(message_user_courant)
            commande = {'model': 'llama3.1', 'messages': messages}

            producer = messages_thread.get_producer()

            # Recuperer le message_id pour faire la correlation
            commande, self.correlation_id = await producer.chiffrer(
                enveloppes, Constantes.KIND_COMMANDE_INTER_MILLEGRILLE, commande, domaine=RELAI_DOMAINE, action='chat')

            message_assistant = ''
            reponse = None
            async for reponse in producer.executer_commande_stream(commande, RELAI_DOMAINE, action='chat',
                                                                   exchange=Constantes.SECURITE_PRIVE, noformat=True):
                parsed = reponse.parsed.copy()
                try:
                    token = parsed['message']['content']
                    message_assistant += token
                    print(token, end='')
                except:
                    logger.debug('Keep alive')

            if reponse is not None:
                message_response = reponse.parsed
                del message_response['__original']
                message_response['message']['content'] = message_assistant  # Concatenate entire response
                messages.append({'role': message_response['message']['role'], 'content': message_response['message']['content']})

            print('\n')  # End


async def callback_reply_q(message: MessageWrapper, messages_module):
    global CHAT_INSTANCE
    global EVENT_WAIT

    logger.info("Message recu : %s" % message)

    if message.kind == Constantes.KIND_REPONSE_CHIFFREE:
        # Reponse
        rk_split = message.routing_key.split('.')
        action = rk_split.pop()
        partition = rk_split.pop()

        if CHAT_INSTANCE and partition == CHAT_INSTANCE.correlation_id:
            await CHAT_INSTANCE.response_q.put(message)
        else:
            content = message.parsed
            del content['__original']
            print("Response\n%s" % json.dumps(content, indent=2))
            EVENT_WAIT.set()

    else:
        logger.info("Message recu : ", message.contenu)
        # partition = message.routage['partition']
        # action = message.routage['action']


def get_maitredescles_certificates() -> list[EnveloppeCertificat]:
    fiche_response = requests.get('http://localhost/fiche.json')
    fiche = fiche_response.json()

    contenu_fiche = json.loads(fiche['contenu'])

    enveloppes = list()
    for cert_chiffrage in contenu_fiche['chiffrage']:
        enveloppe = EnveloppeCertificat.from_pem('\n'.join(cert_chiffrage))
        enveloppes.append(enveloppe)

    if len(enveloppes) == 0:
        raise Exception('Maitre des cles certificate is not available')

    return enveloppes


if __name__ == '__main__':
    logging.basicConfig(format=LOGGING_FORMAT, level=logging.WARN)
    level = logging.ERROR
    logging.getLogger(__name__).setLevel(level)
    logging.getLogger('millegrilles_messages').setLevel(level)
    logging.getLogger('millegrilles_ollama_relai').setLevel(level)

    asyncio.run(main())
