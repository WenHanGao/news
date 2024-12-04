import requests
import json

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, Union
from rich import print as pprint

BASE_URL = "http://10.230.252.6:11434"
API_URL = f"{BASE_URL}/api/generate"


def request_llm(prompt: str, only_text: bool = True) -> Union[str, Dict]:
    body = {
        "model": "llama3.1:70b-instruct-q4_0",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }

    response = requests.post(url=API_URL, json=body)
    response_json = json.loads(response.text)
    if only_text:
        return response_json["response"]
    return response_json


def langchain_llm(output_class: BaseModel, query: str) -> BaseModel:
    model = OllamaLLM(
        base_url=BASE_URL,
        model="llama3.1:70b-instruct-q4_0",
        temperature=0.0,
    )

    # # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=output_class)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    prompt_and_model = prompt | model
    output = prompt_and_model.invoke({"query": query})
    return parser.invoke(output)


class RelevantIndicator(BaseModel):
    """A class to indicate if a news passage is about causes to credit score changes of a company"""

    is_relevant: bool = Field(
        ...,
        description="true if the passage mentions causes to credit score changes, else false",
    )
    content: str = Field(..., description="original content of the news passage")


class Cause(BaseModel):
    """A class for the cause of credit score changes sythesized from news passage"""

    cause: str = Field(..., description="the cause for the credit score change")
    category: str = Field(..., description="the category asscociated with the cause")


if __name__ == "__main__":
    import os
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    FILE_PATH = os.path.abspath(__file__)
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(FILE_PATH)))

    with open(
        os.path.join(
            BASE_PATH, "data", "sample_dummies", "sample_credit_downgrade.txt"
        ),
        "r",
    ) as f:
        data = f.readlines()
    raw_news = "".join(data)

    doc = Document(
        page_content=raw_news,
        metadata={"source": "sample"},
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents([doc])

    for split in splits:
        query = f"""
        You are an expert in credit analysis.
        Does the following passage mentions the causes of credit score changes of SoftBank explicitly?

        Passage:
        {split.page_content}

        Output: """
        response = langchain_llm(output_class=RelevantIndicator, query=query)

        if response.is_relevant:
            query = f"""
            You are an expert in credit analysis.
            Following passage mentions the causes of credit score changes of SoftBank.
            Sythesis the cause in a short sentence and associate it with a category.

            Passage:
            {split.page_content}

            Output: """
            response = langchain_llm(output_class=Cause, query=query)
            print(split.page_content)
            pprint(response.model_dump())
