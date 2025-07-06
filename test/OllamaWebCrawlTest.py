import asyncio
import json
import urllib.parse
import requests
import tiktoken

from bs4 import BeautifulSoup
from ollama import AsyncClient, GenerateResponse
from textwrap import wrap

from millegrilles_ollama_relai.OllamaKnowledgeBase import KnowledgBaseHandler
from millegrilles_ollama_relai.Structs import SummaryKeywords, LinkIdPicker, KnowledgeBaseSearchResponse
from millegrilles_ollama_relai.prompts.knowledge_base_prompt import KNOWLEDGE_BASE_INITIAL_SUMMARY_PROMPT, \
    KNOWLEDGE_BASE_FIND_PAGE_PROMPT, KNOWLEDGE_BASE_SYSTEM_USE_ARTICLE_PROMPT

# CONST_MODEL = "deepseek-r1:8b-0528-qwen3-q8_0"
# CONST_MODEL = "gemma3:27b-it-qat-LOPT"
# CONST_MODEL = "gemma3:12b-it-qat-LOPT"
# CONST_MODEL = "gemma3n:e2b-it-q4_K_M"
CONST_MODEL = "gemma3n:e2b-it-q4_K_M-LOPT"
# CONST_MODEL = "gemma3n:e4b-it-q8_0-LOPT"
# CONST_MODEL = "gemma3n:e4b-it-q8_0"
# CONST_MODEL = "gemma3n:e4b-it-fp16-LOPT"
# CONST_MODEL = "llama3.2:3b-instruct-q8_0"

CONST_SUMMARY_NUM_PREDICT_KEYWORDS = 512
CONST_SUMMARY_NUM_PREDICT_RESPONSE = 2048
CONST_THINK = None

# CONST_CONTEXT_LEN = 8192
CONST_CONTEXT_LEN = 12288
CONST_LIMIT_ARTICLE = int(2.5 * CONST_CONTEXT_LEN)
CONST_HOSTNAME = "https://libs.millegrilles.com"


# async def crawl_search(topic: str):
#     params = [
#         "books.name=wikipedia_en_all_maxi_2023-11",
#         # f"pattern={urllib.parse.quote_plus(", ".join(keywords))}",
#         f"pattern={urllib.parse.quote_plus(topic)}",
#         "userlang=en",
#         "start=1",
#         "pageLength=30",
#     ]
#     params = "&".join(params)
#     search_url = f"{CONST_HOSTNAME}/kiwix/search?{params}"
#
#     response = await asyncio.to_thread(requests.get, search_url)
#     response.raise_for_status()
#
#     data = response.text
#     soup = BeautifulSoup(data, "html.parser")
#     results = soup.select_one(".results")
#
#     items = results.select("li")
#
#     links = list()
#
#     print("Search results")
#     item_id = 1
#     for item in items:
#         anchor = item.select_one("a")
#         url = anchor.attrs.get("href")
#         title = anchor.text.strip()
#         description = item.select_one("cite").text[:100]
#         print(f"{item_id}. {title}: {CONST_HOSTNAME}{url}")
#         print(description)
#         links.append({"linkId": item_id, "title": title, "description": description, "url": url})
#         item_id += 1
#     print("--------------")
#
#     return search_url, links

# async def crawl_get_page(client: AsyncClient, topic: str, search_results: list[dict]):
#
#     # Extract the title, description and linkId to put in the context
#     mapped_results = [{
#         'link_id': r['linkId'],
#         'title': r['title'],
#         # 'description': r['description'],
#     } for r in search_results]
#
#     prompt = f"""
# <topic>
# {topic}
# </topic>
# <searchResults>
# {json.dumps(mapped_results)}
# </searchResults>
# """
#
#     print(f"crawl_get_page prompt size: {len(prompt)}, system prompt size: {len(KNOWLEDGE_BASE_FIND_PAGE_PROMPT)}")
#
#     output = await client.generate(
#         model=CONST_MODEL,
#         system=KNOWLEDGE_BASE_FIND_PAGE_PROMPT,
#         prompt=prompt,
#         think=CONST_THINK,
#         format=LinkIdPicker.model_json_schema(),
#         options={"num_predict": CONST_SUMMARY_NUM_PREDICT_KEYWORDS, "temperature": 0.01},
#     )
#
#     # Fetch the selected link by linkId
#     print(f"Chosen search result: {output.response}")
#     response_dict = LinkIdPicker.model_validate_json(output.response)
#     link_id = response_dict.link_id
#     chosen_link = [l for l in search_results if l['linkId'] == link_id].pop()
#     print(f"Chosen page: {chosen_link['title']}\n{chosen_link['description']}\n{chosen_link['url']}")
#     page_url = chosen_link['url']
#     url = f"{CONST_HOSTNAME}{page_url}"
#
#     response = await asyncio.to_thread(requests.get, url)
#     response.raise_for_status()
#
#     data = response.text
#     soup = BeautifulSoup(data, "html.parser")
#     content = soup.select_one("#mw-content-text")
#     content_string = content.text
#
#     return url, content_string

# async def review_article(client: AsyncClient, user_prompt: str, language: str, assistant_response: str, article: str):
#
#     article_truncated = article[:CONST_LIMIT_ARTICLE]
#
#     prompt = f"""
# <userProfile>
# user_language: {language}
# </userProfile>
#
# <query>
# {user_prompt}
# </query>
#
# <valueToCheck>
# {assistant_response}
# </valueToCheck>
#
# <reference>
# {article_truncated}
# </reference>
#
# <instructions>
# Produce the summary of reference and the fact check of valueToCheck as per the instructions in the system prompt.
# </instructions>
# """
#
#     # print(article_truncated)
#     token_len = check_token_len(prompt+KNOWLEDGE_BASE_SYSTEM_USE_ARTICLE_PROMPT)
#     print(f"Prompt len: {len(prompt)}, System prompt len: {len(KNOWLEDGE_BASE_SYSTEM_USE_ARTICLE_PROMPT)}, Total token len: {token_len}")
#     #if token_len > CONST_CONTEXT_LEN - 700:  # Reserve 700 token for model template (not exact)
#     #    raise Exception("Prompt too long")
#
#     stream = await client.generate(
#         model=CONST_MODEL,
#         system=KNOWLEDGE_BASE_SYSTEM_USE_ARTICLE_PROMPT,
#         think=CONST_THINK,
#         prompt=prompt,
#         stream=True,
#         options={"num_predict": CONST_SUMMARY_NUM_PREDICT_RESPONSE, "temperature": 0.01},
#     )
#
#     async for value in stream:
#         yield value

# async def query_with_keywords(client: AsyncClient, prompt: str):
#     output = await client.generate(
#         model=CONST_MODEL,
#         system=KNOWLEDGE_BASE_INITIAL_SUMMARY_PROMPT,
#         prompt=prompt,
#         think=CONST_THINK,
#         format=SummaryKeywords.model_json_schema(),
#         options={"num_predict": CONST_SUMMARY_NUM_PREDICT_KEYWORDS, "temperature": 0.01},
#     )
#
#     print(f"Summary/Keywords: {output.response}")
#     response_dict = SummaryKeywords.model_validate_json(output.response)
#
#     assistant_response = response_dict.s
#     print(f"Initial response\n{assistant_response}\n")
#
#     topic = response_dict.t
#     if not topic:
#         print("*** Response done ***")
#         return  # Done
#
#     language = response_dict.l
#
#     # keywords = response_dict.kw
#     # if not keywords or len(keywords) == 0:
#     #     print("*** Response done ***")
#     #     return  # Done
#
#     # search_result = await crawl_search(keywords)
#     search_url, search_result = await crawl_search(topic)
#     if len(search_result) == 0:
#         print("No search results identified\n*** Response done ***")
#         return  # Done
#
#     url, article = await crawl_get_page(client, topic, search_result)
#
#     print(f"Topic: {topic}")
#     print(f"Search page: {search_url}")
#     print(f"Verifying response with url: {url}")
#
#     flag_init = False
#     async for chunk in review_article(client, prompt, language, assistant_response, article):
#         if not flag_init:
#             print("*** Response ***\n")
#             flag_init = True
#         if chunk.done:
#             print(f"\n\nPrompt eval count: {chunk.prompt_eval_count}")
#             if chunk.prompt_eval_count >= CONST_CONTEXT_LEN:
#                 raise Exception("Context window excedeed")
#         print(chunk.response, end='')
#     print("\n*** Response done ***")

async def main():

    # prompt = "Where did the St. Lawrence river in Canada get its name from?"
    # prompt = "D'où vient le nom Saint-Laurent pour le fleuve canadien?"
    # prompt = "What was the cause of the French Revolution?"
    prompt = "Quelle était la cause de la révolution française?"
    # prompt = "What is retrieval augmented generation (RAG)?"
    # prompt = "When did aliens invade earth?"
    # prompt = "Are UFO encounters considered real or is it a hoax?"
    # prompt = "Hi"
    # prompt = "What is your cutoff date?"
    # prompt = "What is taxonomy in biology?"
    # prompt = "Are there countries with no military capability?"
    # prompt = "Is Saint Lawrence the patron saint of sailors and merchants?"
    # prompt = "What was the state of Iceland during the cold war?"
    # prompt = "Qu'est-ce qu'un coureur des bois?"
    # prompt = "How was lithography invented to make computer chips?"

    client = AsyncClient()

    knowledge_base_handler = KnowledgBaseHandler(client)
    async for chunk in knowledge_base_handler.run_query(prompt):
        if chunk.complete_block:
            print_text(chunk.text)
        else:
            print(chunk.text, end='')
    print('\n\n*** Response done ***')
    # await query_with_keywords(prompt)

def check_token_len(prompt: str):
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    content_len = len(encoding.encode(prompt))
    return content_len

def print_text(text: str):
    for chunk in wrap(text, width=120):
        print(chunk)

if __name__ == '__main__':
    asyncio.run(main())
