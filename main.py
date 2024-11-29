from fastapi import FastAPI
from langserve import add_routes
import uvicorn
from typing import Any, List, Union
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from agent import agent_executor
from agent import agent_executor2
import os


class Input(BaseModel):
    input: str



class Output(BaseModel):
    output: Any


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/invoke",
)

add_routes(
    app,
    agent_executor2.with_types(input_type=Input, output_type=Output),
    path="/chatbot",
)

if __name__ == "__main__":

    uvicorn.run(app)

    # host="0.0.0.0", port=8002