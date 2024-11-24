import requests
from langchain_core.messages import HumanMessage, AIMessage

inputs = {"input": {"input": "what is the length of the word audioee?", "chat_history": []}}
response = requests.post("http://localhost:8000/invoke", json=inputs)

response.json()


chat_history = []

while True:
    human = input("Human (Q/q to quit): ")
    if human in {"q", "Q"}:
        print('AI: Bye bye human')
        break

    ai = None
    print("AI: ")
    async for chunk in remote_runnable.astream({"input": human, "chat_history": chat_history}):
        # Agent Action
        if "actions" in chunk:
            for action in chunk["actions"]:
                print(
                    f"Calling Tool ```{action['tool']}``` with input ```{action['tool_input']}```"
                )
        # Observation
        elif "steps" in chunk:
            for step in chunk["steps"]:
                print(f"Got result: ```{step['observation']}```")
        # Final result
        elif "output" in chunk:
            print(chunk['output'])
            ai = AIMessage(content=chunk['output'])
        else:
            raise ValueError
        print("------")        
    chat_history.extend([HumanMessage(content=human), ai])