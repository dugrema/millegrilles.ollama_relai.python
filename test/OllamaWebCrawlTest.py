import asyncio
import json
import urllib.parse
import requests
import ollama
from bs4 import BeautifulSoup
from ollama import AsyncClient

# CONST_MODEL = "deepseek-r1:8b-0528-qwen3-q8_0"
# CONST_MODEL = "gemma3:27b-it-qat-LOPT"
# CONST_MODEL = "gemma3:12b-it-qat-LOPT"
# CONST_MODEL = "gemma3n:e4b-it-q8_0-LOPT"
CONST_MODEL = "gemma3n:e2b-it-q4_K_M-LOPT"
# CONST_MODEL = "llama3.2:3b-instruct-q8_0"

CONST_SUMMARY_NUM_PREDICT_KEYWORDS = 100
CONST_SUMMARY_NUM_PREDICT_RESPONSE = 1536
CONST_THINK = None

CONST_LIMIT_ARTICLE = 20_000

SYSTEM_KEYWORDS = """
You are an AI assistant that generates keywords to find relevant information on a local instance of Wikipedia.

# Instructions

* Generate a few keywords in English to apply to a local search engine.
* Each keyword should be a single word unless it is a name.
** Do not translate the name of people, use it as provided.
** If the keyword is a name in another language for a place, technology, geographic feature, etc. then translate it to English.
* Each keyword *must* be directly related to the user query.
* Return the keywords in JSON format: {"keywords": ["word1", "word2", "word3"]}.
* Do not use markdown formatting. You must output plain JSON. 
"""

SYSTEM_FIND_PAGE = """
You are an AI assistant that identifies a wikipedia page to use for fact-checking a user prompt.
You are provided with a user prompt and a list of search results.

# Instructions

* Check the user prompt.
* From the list of search results, identify a URL that most closely matches the user query.
* Return that url using JSON in the form of: {"url": "/kiwix/content/wikipedia_en_all_maxi_2023-11/A/XXXXXXXXX"}
* Make sure to *copy the entire* "href" element properly, do not truncate the url.
* Do not use markdown formatting. You must output plain JSON.
"""

SYSTEM_USE_ARTICLE = """
You are an AI assistant that answers user questions factually. Since you have a cutoff date and your information may
also be incomplete, a wikipedia article was selected to assist you in providing the most factual and up to date 
information.

# Instructions

* Answer the user's query directly factually with your knowledge.
* Use the user's language in your answer.
* Make sure to answer the user's question immediately and directly, do not summarize the article.
* The article's information is authoritative on its topic, it's main purpose is to fact-check your knowledge on that topic.
* If any of the article is misaligned or contradicts your own information, you may use information from the article as a source for your answer.
* Only if the article does not correspond to the user's question or intent, you must answer with your own knowledge.
"""

async def crawl_search(keywords: list[str]):
    params = [
        "books.name=wikipedia_en_all_maxi_2023-11",
        f"pattern={urllib.parse.quote_plus(" ".join(keywords))}",
        "userlang=en"
    ]
    params = "&".join(params)
    url = f"https://libs.millegrilles.com/kiwix/search?{params}"

    response = await asyncio.to_thread(requests.get, url)
    response.raise_for_status()

    data = response.text
    soup = BeautifulSoup(data, "html.parser")
    results = soup.select_one(".results")
    results_html = str(results)

    print(f"HTML result page\n{results_html}")

    return results_html

async def crawl_get_page(client: AsyncClient, user_prompt: str, search_results: str):
    prompt = f"""
<query>
{user_prompt}
</query>

<searchResults>
{search_results}
</searchResults>
"""

    output = await client.generate(
        model=CONST_MODEL,
        system=SYSTEM_FIND_PAGE,
        prompt=prompt,
        think=CONST_THINK,
        options={"num_predict": CONST_SUMMARY_NUM_PREDICT_KEYWORDS, "temperature": 0.01},
    )

    print(f"Chosen page: {output.response}")
    response_dict = json.loads(output.response)
    page_url = response_dict['url']
    if not page_url.startswith('/'):
        page_url = f"/{page_url}"
    page_url = page_url.replace('/kiwix/content/wiki/', '/kiwix/content/wikipedia_en_all_maxi_2023-11/')
    url = f"https://libs.millegrilles.com{page_url}"

    response = await asyncio.to_thread(requests.get, url)
    response.raise_for_status()

    data = response.text
    soup = BeautifulSoup(data, "html.parser")
    content = soup.select_one("#mw-content-text")
    content_string = content.text

    return content_string

async def review_article(client: AsyncClient, user_prompt: str, article: str):

    article_truncated = article[:CONST_LIMIT_ARTICLE]

    prompt = f"""
<userProfile>
   user_language: en_CA
</userProfile>
<query>
{user_prompt}
</query>
<article>
{article_truncated}
</article>
    """

    print(article_truncated)
    print(f"Prompt len: {len(prompt)}, System prompt len: {len(SYSTEM_USE_ARTICLE)}")

    stream = await client.generate(
        model=CONST_MODEL,
        system=SYSTEM_USE_ARTICLE,
        think=CONST_THINK,
        prompt=prompt,
        stream=True,
        options={"num_predict": CONST_SUMMARY_NUM_PREDICT_RESPONSE, "temperature": 0.01},
    )

    async for value in stream:
        yield value

async def query_with_keywords():

    client = AsyncClient()

    prompt = "Where did the St. Lawrence river in Canada get its name from?"
    # prompt = "D'où vient le nom Saint-Laurent pour le fleuve canadien?"
    # prompt = "What was the cause of the French Revolution?"
    # prompt = "What is retrieval augmented generation (RAG)?"
    # prompt = "When did aliens invade earth?"
    # prompt = "Are UFO encounters considered real or is it a hoax?"

    output = await client.generate(
        model=CONST_MODEL,
        system=SYSTEM_KEYWORDS,
        prompt=prompt,
        think=CONST_THINK,
        options={"num_predict": CONST_SUMMARY_NUM_PREDICT_KEYWORDS, "temperature": 0.01},
    )

    print(f"Keywords: {output.response}")
    response_dict = json.loads(output.response)
    keywords = response_dict['keywords']
    search_result = await crawl_search(keywords)
    article = await crawl_get_page(client, prompt, search_result)

    flag_init = False
    async for chunk in review_article(client, prompt, article):
        if not flag_init:
            print("*** Response ***\n")
            flag_init = True
        print(chunk.response, end='')
    print("\n*** Response done ***")

async def main():
    await query_with_keywords()


if __name__ == '__main__':
    asyncio.run(main())
