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


review_prompt = """
    Ti verrà fornita la recensione di un prodotto scritta da un cliente.
    Il tuo obiettivo è analizzarla seguendo lo schema fornito.
    - product: nome del prodotto recensito
    - sentiment: giudizio generale della recensione
    - rating: voto da 1 a 5 che secondo te il cliente darebbe
    - pros: elenco dei punti di forza citati
    - cons: elenco dei difetti citati
    - summary: riassunto della recensione in una frase
"""


class Sentiment(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class ReviewAnalysis(BaseModel):
    product: str
    sentiment: Sentiment
    rating: int
    pros: list[str]
    cons: list[str]
    summary: str


def analyze_review(text):
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": dedent(review_prompt)},
            {"role": "user", "content": text},
        ],
        response_format=ReviewAnalysis,
    )
    return completion.choices[0].message


def esempio_1():
    print("ESEMPIO 1: analisi di recensioni\n")
    reviews = [
        "Ho comprato le cuffie Sonic X3 un mese fa. Il suono è fantastico e la batteria dura tutto il giorno, "
        "però sono scomode dopo un'ora e il bluetooth ogni tanto si disconnette.",
        "L'aspirapolvere Cleanix è un disastro: fa un rumore pazzesco, si è rotto dopo due settimane "
        "e l'assistenza non risponde. Sconsigliato.",
    ]

    for text in reviews:
        message = analyze_review(text)

        if message.refusal:
            print("Richiesta rifiutata:", message.refusal)
            continue

        result = message.parsed
        print(f"Prodotto: {result.product}")
        print(f"Sentiment: {result.sentiment.value}")
        print(f"Voto: {result.rating}/5")
        print("Pro:")
        for p in result.pros:
            print(f"- {p}")
        print("Contro:")
        for c in result.cons:
            print(f"- {c}")
        print(f"Riassunto: {result.summary}\n")


restaurant_prompt = """
    Sei un assistente per la prenotazione di ristoranti.
    Ti viene fornita la richiesta di un utente e un contesto aggiuntivo.
    Hai a disposizione uno strumento per cercare un ristorante in un database.
    In base alla richiesta e al contesto, determina i valori più probabili
    dei parametri da usare per la ricerca.
"""


class Cuisine(str, Enum):
    italian = "italian"
    japanese = "japanese"
    mexican = "mexican"
    indian = "indian"
    vegetarian = "vegetarian"


class RestaurantSearchParameters(BaseModel):
    city: str
    cuisine: Cuisine
    people: int
    time: str


def search_restaurant(user_input, context):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": dedent(restaurant_prompt)},
            {"role": "user", "content": f"CONTEXT: {context}\n USER INPUT: {user_input}"},
        ],
        tools=[
            openai.pydantic_function_tool(
                RestaurantSearchParameters,
                name="restaurant_search",
                description="Search a restaurant in the database",
            )
        ],
    )
    return response.choices[0].message.tool_calls


def esempio_2():
    print("ESEMPIO 2: ricerca ristoranti\n")
    richieste = [
        {
            "user_input": "Vorrei portare i miei genitori a mangiare il sushi sabato sera verso le 20.",
            "context": "City: Milano, Group: 3 people",
        },
        {
            "user_input": "Cerco un posto per una cena di compleanno con gli amici, ci piacciono i tacos.",
            "context": "City: Roma, Group: 8 people, Time: 21:00",
        },
        {
            "user_input": "Pranzo di lavoro domani, uno dei colleghi non mangia carne.",
            "context": "City: Torino, Group: 4 people, Time: 13:00",
        },
    ]

    for r in richieste:
        tool_calls = search_restaurant(r["user_input"], r["context"])
        args = json.loads(tool_calls[0].function.arguments)
        print(f"Input: {r['user_input']}")
        print(f"Context: {r['context']}")
        for key, value in args.items():
            print(f"{key}: '{value}'")
        print()


if __name__ == "__main__":
    esempio_1()
    esempio_2()
