PROMPT_INDEXING_SYSTEM_DOCUMENT = """
# Task

Summarize the content in the Document tag. Your output must use the User language provided below.

# Personalization

User language: {language}

# Information required on all types

* Respond in json, do not use markdown. 
* Do not ask questions, just provide the information. This is not an interactive prompt.
* Summary in the user's language.
* Create a list of keywords / tags in the user's language.
* Output in the user's own language, the user's language provided in the UserProfile tag. 
* When possible, include the number of pages at the end of the summary.

# Summary guide depending on document type

Detect the type of document according to the content that was provided.

* If the document is an invoice, summarize as: Invoice of company "INSERT COMPANY NAME", total amount:"INSERT AMOUNT" due by "INSERT DATE".
* If the document is a contract, summarize as: Contract between parties "PARTY A", "PARTY B" and "PARTY C" on "TOPIC OF CONTRACT". Also mention contractual dates when possible.
* If the document is an article, include the *title* and then a *summary* using between 75 and 200 words. 
  For example: An article title. A new type of quantum state has been uncovered by scientists at a restaurant ...
"""

PROMPT_INDEXING_SYSTEM_IMAGES = """
# Task

Generate a detailed description of the image. Your output must use the User language provided below.

# Personalization

User language: {language}

## Instructions

* Do not ask questions, just provide the information. This is not an interactive prompt.
* Respond in json, do not use markdown.
* Generate a detailed description in the summary field.
* Provide a list of keywords / tags.
* Use the user language provided to respond.

"""
