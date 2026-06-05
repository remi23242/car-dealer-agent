import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

ANSWER_SYSTEM = """\
You are a concise car dealership assistant answering on a voice call.
Use ONLY the inventory results below to answer the user's specific question.
If the results do not contain information relevant to the question, say so in one sentence and stop — do NOT list cars.
If the results ARE relevant, mention the car name, price, and one key spec. 2 sentences max. No markdown. No lists."""

FAKE_INVENTORY = """
Make: Honda, Model: Prelude, Year: 2000, Price: 3843, Transmission: manual
Make: Toyota, Model: ECHO, Year: 2003, Price: 10775, Transmission: manual
Make: Toyota, Model: ECHO, Year: 2004, Price: 10885, Transmission: manual
"""

TESTS = [
    "Could you tell me more about speakers?",
    "Do you have any Toyota cars?",
    "What is the price of the Honda Prelude?",
    "Tell me about your SUVs",
]


async def main() -> None:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    for query in TESTS:
        resp = await llm.ainvoke([
            SystemMessage(content=f"{ANSWER_SYSTEM}\n\nINVENTORY:\n{FAKE_INVENTORY}"),
            HumanMessage(content=query),
        ])
        print(f"Q: {query}")
        print(f"A: {resp.content.strip()}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
