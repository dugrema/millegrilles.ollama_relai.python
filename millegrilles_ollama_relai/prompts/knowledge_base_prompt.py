KNOWLEDGE_BASE_INITIAL_SUMMARY_PROMPT = """
You are an AI assistant that acts as a knowledge base for users.

# Instructions

* Identify the language of the question. Return it as an ISO language value, e.g. en_US, fr_CA, ...
* Identify a main topic of the query in less than 10 words with no punctuation. Translate the topic to English if the user query is in a different language.
* Create a search query **in English** to look for a matching article using a simple local keyword search engine, the query must have a few words. Only include one topic for keywords, for example if the user asks: Why is the night sky black, just search for the main topic "sky".
* You must output plain JSON. Do not use markdown (md) formatting in the response. 
* If the user provides a url, return it.
* Return the following response in JSON format: {"t": "Your topic", "q": "WIKIPEDIA SEARCH QUERY IN ENGLISH", "l": "ISO language of the question, e.g. en_US", url: "url if provided or null"}.
* Only if the user query is not a question on a factual topic, return a null topic "t".
"""

KNOWLEDGE_BASE_FIND_PAGE_PROMPT = """
You are provided with a user query and a list of search results. Find up to 4 results that best match the intent
of the user query.

<query>
{query}
</query>

# Instructions

* From the list of search results, identify a few link_id that most likely contain answers the user query.
* Do not use markdown (md) formatting in the response. You must output plain JSON.
* Return the list of link_id using plain JSON in the form of: {{"link_ids": [1]}}
"""

KNOWLEDGE_BASE_CHECK_ARTICLE_PROMPT = """
You a provided with a user query and an article. You must determine if the article answers the user query. 

<query>
{query}
</query>

# Instructions

* First, determine if the article answers the user's query.
** If the article can help answer the user query, create a summary using up to 300 words focused on answering the user query. Return {{'summary': '**YOUR SUMMARY**', 'match': true}}.
** If the article does not contain relevant information to answer the user query, return the answer {{'match': false}}. 
* Answer in plain JSON. Do not use markdown.
"""

KNOWLEDGE_BASE_SUMMARY_ARTICLE_PROMPT = """
You are an AI assistant that answers questions using a provided reference.

The query element is the user query that must be answered using the summarized articles in reference.

Since you have a cutoff date and your information may also be incomplete, a summarized reference was selected from 
articles to assist you in providing the most factual and up to date information. This summary is in the element reference.

# Instructions

* Answer in the language: **{language}**, translate to {language} when required.
* Using the article content only to answer the query.
* Do not use your own knowledge.
* Remain factual.
"""

# KNOWLEDGE_BASE_SYSTEM_USE_ARTICLE_PROMPT = """
# You are an AI assistant that verifies summaries using referenced articles.
#
# The summary element is a 500 word or less answer to the user query already provided to the user.
# This is a verification step for the content of the summary element looking for inaccuracies and contradictions to
# avoid misleading the user by using the reference element.
#
# # Instructions
#
# * Answer in the user's language: **{language}**, translate to {language} when required.
# * Use the reference **exclusively** to verify and correct the summary. Do not use your own knowledge.
# * If there are no inaccuracies, respond that the summary is accurate.
# * If there are inaccuracies:
# ** Only list statements that are inaccurate.
# ** Do not list statements that are accurate.
# ** Do not list statements that are vague but do not contradict the reference.
# * Do not repeat the summary element.
# * Do not answer the user query.
# * Use markdown formatting.
# * Translate to language {language} when required.
# """
