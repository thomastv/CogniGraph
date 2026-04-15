import json
import logging

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from cognigraph.db import save_preference


def extract_and_save_preference(llm, user_message: str) -> None:
    """Extract one stable preference/fact from user input and persist it."""
    logging.info("Running preference extraction")

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract stable user preferences/facts from the user message. "
                "If present, return ONLY JSON in this format: {\"key\":\"snake_case_key\",\"value\":\"value\"}. "
                "If nothing is worth saving, return {}.",
            ),
            ("human", "User message: {message}"),
        ]
    )

    chain = extraction_prompt | llm | StrOutputParser()
    raw_output = chain.invoke({"message": user_message})

    cleaned_output = raw_output.strip()
    if "```json" in cleaned_output:
        cleaned_output = cleaned_output.split("```json", 1)[1].split("```", 1)[0].strip()
    elif cleaned_output.startswith("```"):
        cleaned_output = cleaned_output.split("```", 1)[1].rsplit("```", 1)[0].strip()

    try:
        extracted = json.loads(cleaned_output)
        if isinstance(extracted, dict) and extracted.get("key") and extracted.get("value"):
            key = str(extracted["key"]).strip()
            value = str(extracted["value"]).strip()
            if key and value:
                save_preference(key, value)
                logging.info(f"Saved user preference: {key}={value}")
    except json.JSONDecodeError:
        logging.info("No valid preference JSON extracted from this message")
