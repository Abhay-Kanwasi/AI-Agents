import json
import operator
import traceback
from groq import Groq
from uuid import uuid4
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Any
from langchain.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage

import logging
logger = logging.getLogger(__name__)

api_key_groq = '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'

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    default_prompt: str
    final_prompt: str
    fields: Any

class Agent:

    def __init__(self, model, system="") -> None:
        self.system = system
        self.memory_persistance = MemorySaver()
        graph_flow = StateGraph(AgentState)

        graph_flow.add_node("initialize", self.initialize) #1
        graph_flow.add_node("input_requirement", self.input_requirement) #1
        graph_flow.add_node("summarizer", self.summarizer) #2
        graph_flow.add_node("final_summarizer", self.final_summarizer) #4
        graph_flow.add_edge("initialize", "input_requirement")
        graph_flow.add_edge("input_requirement","summarizer")
        graph_flow.add_conditional_edges(
            "summarizer",
            self.decision_review,
            {True: "final_summarizer", False: "input_requirement"}) #3
        graph_flow.add_edge("final_summarizer",END)
        graph_flow.set_entry_point("initialize") # entry point

        self.graph = graph_flow.compile(checkpointer=self.memory_persistance, interrupt_after=["input_requirement"])
        self.model = model
        self.max_tries = 3
        self.counter = 0

    def getChatResponse(self, state):
        print("______Getting inside getChatResponse...")
        inputMsg = ""
        for msg in state['messages']:
            if isinstance(msg, HumanMessage):
                if msg.content.lower() not in inputMsg:
                    inputMsg = inputMsg + " " + msg.content.lower()
        print("######## inputMsg", inputMsg)
        name_desc_pair = ""
        for field in state["fields"]:
            name_desc_pair = name_desc_pair + f"{field['name']}: {field['description']}\n"

        prompt_template = """\
            For the following text, extract the following information:
            If any of the information below is not provided then return "None"
            
            {name_desc_pair}
            text: {text}
    
            {format_instructions}
        """

        response_schema = []
        for field in state["fields"]:
            response_schema.append(ResponseSchema(name=field["name"], description=field["description"]))

        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        format_instructions = output_parser.get_format_instructions()
        print("########", format_instructions)
        prompt = ChatPromptTemplate.from_template(template=prompt_template)

        chat = ChatGroq(model="llama-3.1-70b-versatile", temperature=0,
                        api_key=api_key_groq)
        messages = prompt.format_messages(text=inputMsg, format_instructions=format_instructions, name_desc_pair=name_desc_pair)
        print("messages to chat: ", messages)
        response = chat(messages)
        print("response.content in getChatResponse: ", response.content)
        print("______Getting outside getChatResponse... state['messages']: ", state['messages'], "...... output_parser.parse(response.content): ", output_parser.parse(response.content))
        return output_parser.parse(response.content)

    def initialize(self, state: AgentState):
        print("______Getting inside initialize...")
        human_text = ""
        for msg in state['messages']:
            if isinstance(msg, HumanMessage):
                human_text = msg.content
                break

        if not human_text:
            raise ValueError("No HumanMessage found in the state messages.")

        output_dict = self.getChatResponse(state)
        print("output_dict in initialize: ", output_dict)

        disallowed_terms = ["none", "n/a", "null", "not applicable"]
        for key in output_dict:
            for field in state["fields"]:
                if field["name"] == key and not any(field["value"] is not None and term in field["value"].lower() for term in disallowed_terms):
                    field["value"] = output_dict[key]
        print("initialize ", self.counter, state["fields"])
        print("______Getting outside initialize...")
        return {'fields': state["fields"]}

    def input_requirement(self, state: AgentState):
        print("______Getting inside input_requirement...")
        prompt_request = None
        allNone = True
        for field in state["fields"]:
            if field["value"] is not None or field["value"] != 'None':
                allNone = False
                break
        if allNone:
            prompt_request = state["default_prompt"]
        else:
            for field in state["fields"]:
                if field["value"] is None  or field["value"] == 'None':
                    prompt_request = field["prompt"]
                    break
        if prompt_request is None:
            prompt_request = "done"

        userprompt_response = self.model.chat.completions.create(
            messages=[{"role": "user", "content": prompt_request}],
            model="llama3-8b-8192"
        )
        user_prompt = userprompt_response.choices[0].message.content.strip()
        if prompt_request == 'done':
            state["messages"].append(ToolMessage(content=prompt_request, tool_call_id=str(uuid4())))
        else:
            state["messages"].append(ToolMessage(content=user_prompt, tool_call_id=str(uuid4())))
        print("______Getting outside input_requirement...", prompt_request, state["messages"])
        return {'messages': state['messages']}

    def summarizer(self, state: AgentState):
        print("______Getting inside summarizer...")
        human_text = ""
        for msg in state['messages']:
            if isinstance(msg, HumanMessage):
                human_text = msg.content
                break

        if not human_text:
            raise ValueError("No HumanMessage found in the state messages.")

        output_dict = self.getChatResponse(state)
        disallowed_terms = ["none", "n/a", "null", "not applicable", ""]
        for key in output_dict:
            for field in state["fields"]:
                if field["name"] == key and not any(field["value"] is not None and term in field["value"].lower() for term in disallowed_terms):
                    field["value"] = output_dict[key]

        self.counter += 1
        if self.counter >= self.max_tries:
            for field in state["fields"]:
                if field["value"] is None or field["value"] == 'None':
                    field["value"] = field["default"]

        print("______Getting outside summarizer...")
        return {'fields': state["fields"]}

    def decision_review(self, state: AgentState):
        print("______Getting inside decision_review...")
        result = False
        for msg in state['messages']:
            if isinstance(msg, ToolMessage):
                if msg.content == 'done':
                    result = True
                    break
        print("______Getting outside decision_review...")
        return result

    def final_summarizer(self, state: AgentState):
        print("______Getting inside final_summarizer...")
        model_name = state["fields"]

        if not model_name:
            raise ValueError("Model name is not provided.")

        content = ""
        for field in state["fields"]:
            if field["value"] is not None:
                if content != "":
                    content = content + " and "
                content = content + str(field["name"]) + " " + str(field["value"])
        # Prompt to analyze and suggest GPU model and count
        prompt = (
            f"Given the '{content}', {state['final_prompt']} "
        )

        print("######____Prompt::: ", prompt)

        finalsummary_response = self.model.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192"  # Adjust the model name as needed
        )

        gpu_recommendation = finalsummary_response.choices[0].message.content.strip()
        
        state["messages"].append(ToolMessage(content=gpu_recommendation, tool_call_id=str(uuid4())))
        print("______Getting outside final_summarizer...")
        return {'messages': state['messages']}
    
def process(json_data):
    # load_dotenv()
    model = Groq(api_key=api_key_groq)
    inputs = []
    print("process", json_data["messages"])
    for msg in json_data["messages"]:
        if isinstance(msg, HumanMessage):
            inputs.append(msg)
    thread = {"thread_id": "1"}
    abot = Agent(model, system=json_data["system_prompt"])
    abot.graph.update_state(config={"configurable": thread},
                            values={"messages": inputs, "default_prompt": json_data["default_prompt"],
                                    "final_prompt": json_data["final_prompt"],
                                    "reset_memory": False,
                                    "fields": json_data["fields"]})
    result = abot.graph.invoke({"messages": inputs}, config={"configurable": thread})
    # print("############ process->messages", result)
    outMsg = []
    done = False
    # for msg in result['messages']:
    #     if isinstance(msg, HumanMessage):
    #         inputs.append(msg)
    for msg in result['messages']:
        if isinstance(msg, ToolMessage):
            if msg.content == 'done':
                done = True
                break
            outMsg.append(msg.content)
    print("############ process->messages", result)
    if done:
        outMsg = []
        for event in abot.graph.stream(None, {"configurable": thread}, stream_mode="values"):
            print("############ abot.graph state values", event)
            msg_content = event["messages"][-1].content
            if msg_content not in outMsg and msg_content != "done":
                outMsg.append(msg_content)

    return outMsg[0], done


def getResponse(user_input, json_data):
    try:
        json_data = json.loads(json_data)
    except ValueError:
        trace_back = traceback.format_exc()
        logger.error("getResponse " + str(trace_back))

    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    json_data["messages"] = [HumanMessage(content=input) for input in user_input]

    return process(json_data)

son_data_computerbooking_sample={
    "reset_state": False,
    "messages": [],
    "system_prompt": "You are a smart AI assistant. Given a student question, you respond with relevant information ",
    "default_prompt": "Generate one-line prompt telling the user how you would help them with their questions around University of California at Davis campus services",
    "final_prompt":"Book one computer (PC or Mac) from the following available rooms, each with specific numbers of computers: 15 Olson (10 PC, 2 Mac), 91 Shields (13 PC, 1 Mac), 78 Hutchison (8 PC, 0 Mac), 10 Wellman (12 PC, 3 Mac). Upon successful booking, provide a confirmation alert with the booking details and the ID of the booked computer in the following format: \
    Time: [time of booking], \
    Room: [chosen room], \
    Computer: [PC or Mac], \
    Computer ID: [ID of the booked computer]",
    "fields": [{
                    "name": "date",
                    "value": None,
                    "default": "None",
                    "description": "Extract the date for the booking of computer rooms, return `None` in string format.",
                    "prompt": "Generate a single-line prompt asking the user the date in format mm/dd/yyyy for the booking of computer rooms, using a formal tone, but avoid starting with 'Please provide' or 'What' and no extra punctuation."
                    #"Generate one-line prompt asking the user the name of the customer."
                },{
                    "name": "time",
                    "value": None,
                    "default": "None",
                    "description": "Extract the time for the booking of computer rooms, return `None` in string format.",
                    "prompt": "Generate a single-line prompt asking the user the time in format hh:mm for the booking of computer rooms, using a formal tone, but avoid starting with 'Please provide' or 'What' and no extra punctuation."
                }]
}

def agenticWorkflow(prompt_string, app_config):
    logger.info("Coming inside agenticWorkflow...")
    try:
        response, reset = getResponse(prompt_string, app_config["json_data"])
        print("############## done with getResponse")
        if isinstance(response, str):
            cleaned_response = response.strip('"\'')
            return cleaned_response, reset
        else:
            return str(response), reset
    except Exception as e:
        trace_back = traceback.format_exc()
        logger.error("agenticWorkflow " + str(trace_back))



if __name__ == "__main__":
    prompt_string = ["I want to book a Mac computer on 17/21/2025 at 12:32AM"]
    app_config = {
    "json_data": json.dumps(son_data_computerbooking_sample)}
    result, reset = agenticWorkflow(prompt_string, app_config)
    print(f'Result : {result}\nReset : {reset}')
    if reset:
        string = input(f"{result}: ")
        prompt_string.append(string)
        result, reset = agenticWorkflow(prompt_string, app_config)
        print(f"result:- {result}")
    else:
        print(result)