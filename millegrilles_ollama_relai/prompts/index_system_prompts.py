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
  
# Example answers:

{{"summary": "This is an invoice of company ABC for Person D to the amount of 34.76$ due on 2026-01-01. The invoice details items: - 2 pencils, - 4 erasers, - 1 blackboard. The tax amount in 4.12$.", "tags": ["invoice","school suppy","due in January","2026"]}}
{{"summary": "Un paysage automnal ensoleillé montrant les pentes d'une montage avec des arbres au feuillage colorés.", "tags": ["photo","automne","jour","soleil","arbres"]}}
{{"summary": "Empty document.", "tags": ["empty"]}}
{{"summary": "A breathtaking winter landscape captures a serene sunset over a vast, still lake nestled among snow-capped mountains. In the foreground, a steep, snow-covered slope is textured with shadows, showing small, frost-laden trees and bushes. The middle ground features a large lake that perfectly reflects the vibrant orange and yellow hues of the setting sun. The sun, positioned low on the horizon to the right, creates a brilliant golden path across the water's surface. In the background, a majestic range of rugged, snow-covered mountains stretches across the horizon under a clear sky that transitions from pale blue to warm orange. On the far right, a tall, dark evergreen tree with sparse needles frames the scene, its silhouette contrasting against the glowing sky. The overall atmosphere is peaceful and majestic, highlighting the beauty of a cold, alpine wilderness at dusk.", "tags": ["winter, "sunset", "lake", "mountains", "snow", "landscape", "nature", "golden hour", "reflection", "serene", "peaceful", "alpine", "scenery", "frost", "evergreen tree", "wilderness"]}}
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

# Example answers:

{{"title": "An invoice from company ABC", "summary": "This is an invoice of company ABC for Person D to the amount of 34.76$ due on 2026-01-01. The invoice details items: - 2 pencils, - 4 erasers, - 1 blackboard. The tax amount in 4.12$.", "labels": ["invoice","school suppy","due in January","2026"]}}
{{"title": "Un paysage automnal", "summary": "Un paysage automnal ensoleillé montrant les pentes d'une montage avec des arbres au feuillage colorés.", "labels": ["photo","automne","jour","soleil","arbres"]}}

"""
