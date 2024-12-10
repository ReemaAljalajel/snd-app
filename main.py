from fastapi import FastAPI
from langserve import add_routes
import uvicorn
from typing import Any
from pydantic import BaseModel
from agent import agent_executor
from agent import agent_executor2



# This class Defines the structure of the input expected by the API.
# - The `input` field is a string that represents the user's input query.
class Input(BaseModel):
    input: str


# - This class Defines the structure of the response returned by the API.
# - The `output` field can be of any type, allowing flexibility in responses.
class Output(BaseModel):
    output: Any

## Initializing the FastAPI Application
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Adding Routes for Agent Executors
# Two separate routes are defined for the two executors:

#   1. `/invoke`: Maps to `agent_executor`, primarily for testing.
add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/invoke",
)

#   2. `/chatbot`: Maps to `agent_executor2`, which is deployed in production.
add_routes(
    app,
    agent_executor2.with_types(input_type=Input, output_type=Output),
    path="/chatbot",
)

#   2. `/chatbot`: Maps to `agent_executor2`, which is deployed in production.
add_routes(
    app,
    agent_executor2.with_types(input_type=Input, output_type=Output),
    path="/chatbot2",
)

# Main Entry Point
# - Runs the FastAPI application using `uvicorn`
if __name__ == "__main__":

    uvicorn.run(app)

# Default host and port: host="0.0.0.0" (external access), port=8000.
