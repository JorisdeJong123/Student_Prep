from langchain.prompts import PromptTemplate

prompt_template_samenvatting = """

Je bent een specialist in aanbestedingen. Je bent gespecialiseerd in het schrijven van samenvattingen van aanbestedingen. Je hebt een aanbesteding ontvangen van een klant.

Schrijf een gedetailleerde samenvatting van het volgende document. Dit is een deel van de aanbesteding:


{text}

Zorg ervoor dat je geen belangrijke informatie verliest. Wees zo gedetailleerd mogelijk.

Belangrijke informatie is bijvoorbeeld:

Omvang van de opdracht, gunningscriteria en eisen aan de inschrijving.

Wees erg specificiek in de gunningscriteria, omschrijf precies wat er van de inschrijver verwacht wordt.

Schrijf een samenvatting waarbij de lezer wordt meegenomen door de belangrijkste items uit deze aanbesteding.

Het is bedoeld voor een potentiele inschrijver om in te schatten wat er van hem verwacht wordt, wat de omvang van de opdracht is en de eisen aan de inschrijving.


DETAILED SUMMARY IN ENGLISH:"""
PROMPT_SAMENVATTING = PromptTemplate(template=prompt_template_samenvatting, input_variables=["text"])
refine_template_samenvatting = ('''
    "Je bent een specialist in aanbestedingen. Je bent gespecialiseerd in het schrijven van samenvattingen van aanbestedingen. Je hebt een aanbesteding ontvangen van een klant."
    
    
    "We hebben tot op zekere hoogte een bestaande samenvatting gegeven: {existing_answer}\n"
    "We hebben de mogelijkheid om de bestaande samenvatting te verfijnen"
    "(alleen indien nodig) met wat meer context hieronder.\n"


    "------------\n"
    "{text}\n"
    "------------\n"
    "Verfijn, gezien de nieuwe context, de oorspronkelijke samenvatting in het Nederlands"


Zorg ervoor dat je geen belangrijke informatie verliest. Wees zo gedetailleerd mogelijk.

Schrijf een samenvatting waarbij de lezer wordt meegenomen door de belangrijkste items uit deze aanbesteding.

Het is bedoeld voor een potentiele inschrijver om in te schatten wat er van hem verwacht wordt, wat de omvang van de opdracht is en de eisen aan de inschrijving.

Belangrijke informatie is bijvoorbeeld:

Omvang van de opdracht, gunningscriteria en eisen aan de inschrijving.

Wees erg specificiek in de gunningscriteria, omschrijf precies wat er van de inschrijver verwacht wordt.

    "Als de context niet nuttig is, geeft u de originele samenvatting terug. Zorg ervoor dat je gedetailleerd bent in je samenvatting"
'''
)
REFINE_PROMPT_SAMENVATTING = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_samenvatting,
)

prompt_template_questions = """
Your goal is to prepare a student for their an exam. You do this by asking questions about the text below:

{text}

Create questions that will prepare the student for their exam. Make sure not to lose any important information.

QUESTIONS:
"""
PROMPT_QUESTIONS = PromptTemplate(template=prompt_template_questions, input_variables=["text"])
refine_template_questions = ("""
Your goal is to help a student prepare for an exam.
We have received some questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below
"------------\n"
"{text}\n"
"------------\n"

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions. Make sure to be detailed in your questions.
"""
)
REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)