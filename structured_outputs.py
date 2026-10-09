import os
import json
from enum import Enum
from textwrap import dedent

import openai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(".env")
openai.api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI()
MODEL = "gpt-4o"



math_tutor_prompt = """
    Sei un tutor di matematica.
    Ti verrà fornito un problema matematico,
    e il tuo obiettivo sarà quello di fornire una soluzione
    passo dopo passo, insieme a una risposta finale.
    Per ogni passaggio, fornisci l'output come equazione
    e usa il campo explanation per spiegare il ragionamento.
"""


class MathReasoning(BaseModel):
    class Step(BaseModel):
        explanation: str
        output: str

    steps: list[Step]
    final_answer: str


def get_math_solution(question):
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": dedent(math_tutor_prompt)},
            {"role": "user", "content": question},
        ],
        response_format=MathReasoning,
    )
    return completion.choices[0].message


def esempio_1():
    print("ESEMPIO 1: tutor di matematica\n")
    message = get_math_solution("come posso risolvere 8x + 7 = -23")

    if message.refusal:
        print("Richiesta rifiutata:", message.refusal)
        return

    result = message.parsed
    for i, step in enumerate(result.steps):
        print(f"Step {i + 1}: {step.explanation}")
        print(f"   {step.output}\n")
    print("Risposta finale:", result.final_answer)



product_search_prompt = """
    Sei un personal shopper, specializzato nel trovare l'abbinamento perfetto per un utente.
    Ti viene fornito un input dell'utente e un contesto aggiuntivo,
    come il sesso e la fascia d'età dell'utente e la stagione.
    Hai a disposizione uno strumento per cercare in un database gli abiti
    che corrispondono al profilo e alle preferenze dell'utente.

    Ecco le categorie disponibili sul sito:
    - shoes: boots, sneakers, sandals
    - jackets: winter coats, cardigans, parkas, rain jackets
    - tops: shirts, blouses, t-shirts, crop tops, sweaters
    - bottoms: jeans, skirts, trousers, joggers

    Usa nomi di colori comuni.
"""


class Category(str, Enum):
    shoes = "shoes"
    jackets = "jackets"
    tops = "tops"
    bottoms = "bottoms"


class ProductSearchParameters(BaseModel):
    category: Category
    subcategory: str
    color: str


def get_product_search(user_input, context):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": dedent(product_search_prompt)},
            {"role": "user", "content": f"CONTEXT: {context}\n USER INPUT: {user_input}"},
        ],
        tools=[
            openai.pydantic_function_tool(
                ProductSearchParameters,
                name="product_search",
                description="Search for a match in the product database",
            )
        ],
    )
    return response.choices[0].message.tool_calls


def esempio_2():
    print("\nESEMPIO 2: ricerca prodotti\n")
    richieste = [
        {
            "user_input": "Quest'estate andrò a fare un percorso in Scozia. Ci sarà molta pioggia. Aiutami a trovare qualcosa.",
            "context": "Gender: male, Age group: 30-40",
        },
        {
            "user_input": "Sto cercando di completare un look rock. Mi mancano le scarpe. Qualche suggerimento?",
            "context": "Gender: female, Age group: 20-30",
        },
        {
            "user_input": "Puoi aiutarmi a trovare un vestito per una festa a tema Barbie a luglio?",
            "context": "Gender: female, Age group: 20-30",
        },
    ]

    for r in richieste:
        tool_calls = get_product_search(r["user_input"], r["context"])
        args = json.loads(tool_calls[0].function.arguments)
        print(f"Input: {r['user_input']}")
        print(f"Context: {r['context']}")
        for key, value in args.items():
            print(f"{key}: '{value}'")
        print()


if __name__ == "__main__":
    esempio_1()
    esempio_2()
