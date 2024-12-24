import json
import logging
import operator
import traceback
from groq import Groq
from uuid import uuid4
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Any
from langchain.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

logging.basicConfig(level=logging.INFO)
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
        name_desc_pair = ""
        for field in state["fields"]:
            name_desc_pair += f"{field['name']}: {field['description']}\n"
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
        print("Format Instructions:\n", format_instructions)
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chat = ChatGroq(model="mixtral-8x7b-32768", temperature=0,
                        api_key=api_key_groq)
        messages = prompt.format_messages(text=inputMsg, format_instructions=format_instructions, name_desc_pair=name_desc_pair)

        try:
            print("Formatted Messages:\n", messages)
            response = chat(messages, timeout=10)  
            print("Raw response content:", response.content)
            parsed_response = output_parser.parse(response.content)
            print("Parsed response:", parsed_response)

        except Exception as e:
            print("Error occurred during response parsing:", e)
            parsed_response = {"error": "Parsing failed or incomplete response"}

        print("______Getting outside getChatResponse... state['messages']: ", state['messages'])
        return parsed_response

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
        print("______Getting outside input_requirement...")
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
    
    ##################################
    ## Tools for Hardware Sizing Agent
    ##################################
    
    def calc_generalized_pod_requirements(self, rank):
        print("######## Entering calc_generalized_pod_requirements: Determining resources based on rank... ")

        # Base configuration
        requirements = {
            "compute_nodes_required": 1,
            "pcie_nodes_required": 0,
            "gpus_required": 0,
            "gpu_model": None  # Placeholder for GPU model
        }

        # Define scaling logic based on rank bands
        if rank < 150:
            requirements["compute_nodes_required"] = 1
        elif rank < 250:
            requirements["compute_nodes_required"] = 1
            requirements["pcie_nodes_required"] = 1
            requirements["gpus_required"] = 1
            requirements["gpu_model"] = "Nvidia L40"
        elif rank < 300:
            requirements["compute_nodes_required"] = 1
            requirements["pcie_nodes_required"] = 1
            requirements["gpus_required"] = 2
            requirements["gpu_model"] = "Nvidia L40"
        elif rank < 400:
            requirements["compute_nodes_required"] = 2
            requirements["pcie_nodes_required"] = 2
            requirements["gpus_required"] = 4
            requirements["gpu_model"] = "Nvidia L40"
        elif rank < 500:
            requirements["compute_nodes_required"] = 2
            requirements["pcie_nodes_required"] = 1
            requirements["gpus_required"] = 2
            requirements["gpu_model"] = "Nvidia H100"
        elif rank < 600:
            requirements["compute_nodes_required"] = 2
            requirements["pcie_nodes_required"] = 2
            requirements["gpus_required"] = 4
            requirements["gpu_model"] = "Nvidia H100"
        elif rank < 700:
            requirements["compute_nodes_required"] = 4
            requirements["pcie_nodes_required"] = 3
            requirements["gpus_required"] = 6
            requirements["gpu_model"] = "Nvidia H100"
        else:
            requirements["compute_nodes_required"] = 4
            requirements["pcie_nodes_required"] = 4
            requirements["gpus_required"] = 8
            requirements["gpu_model"] = "Nvidia H100"

        # Prepare the result dictionary
        result = {
            "compute_nodes_required": f"{requirements['compute_nodes_required']} Cisco UCS X210c M7",
            "pcie_nodes_required": f"{requirements['pcie_nodes_required']} Cisco UCS X440p",
            "gpus_required": f"{requirements['gpus_required']} {requirements['gpu_model']}" if requirements['gpu_model'] else "None"
        }

        # Output the calculated specifications
        print("#### Calculated Hardware Requirements Based on Generalized Formula:")
        print(f"Compute Nodes: {result['compute_nodes_required']}")
        print(f"PCIe Nodes: {result['pcie_nodes_required']}")
        print(f"GPUs: {result['gpus_required']}")

        return result

    def get_parameter_value(self, model_parameters):
        """Convert model_parameters to a numerical value for ranking."""
        model_parameters = model_parameters.lower()
        if model_parameters.endswith('b'):
            return int(model_parameters[:-1]) * 10**9
        else:
            raise ValueError("Model parameters must be in the form of a number followed by 'b', e.g., '13b'.")
        
    def parse_latency(self, latency_str):
        """Extracts the numerical value from a latency string."""
        if latency_str.endswith('ms'):
            return int(latency_str[:-2])  # Convert '5ms' to 5
        elif latency_str.isdigit():  # No unit provided, assume it's in milliseconds
            return int(latency_str)
        raise ValueError("Invalid latency format. Expected format: '5ms'.")
    
    def parse_data_size(self, data_size_value):
        """Parses data size strings like '5GB' or '2TB' into numerical values in GB."""
        if 'TB' in data_size_value.upper():
            return float(data_size_value.upper().replace('TB', '').strip()) * 1000
        elif 'GB' in data_size_value.upper():
            return float(data_size_value.upper().replace('GB', '').strip())
        else:
            raise ValueError("Data size value must specify units (GB or TB).")

    def selectHardwareSizingPod(self, model_name, model_parameters, rps, latency, data_size_value, data_type_value):
        print("#### coming inside selectHardwareSizingPod", model_parameters, rps, latency, data_size_value, data_type_value)
        # Define ranking scores for model parameters
        model_param_scores = {
            (0, 1 * 10**9): 30,          # < 1b
            (1 * 10**9, 3 * 10**9): 50,  # 1b to 3b
            (3 * 10**9, 7 * 10**9): 80,  # 3b to 7b
            (7 * 10**9, 10 * 10**9): 100,  # 7b to 10b
            (10 * 10**9, 20 * 10**9): 150,  # 10b to 20b
            (20 * 10**9, 40 * 10**9): 250,  # 20b to 40b
            (40 * 10**9, 70 * 10**9): 300,  # 40b to 70b
            (70 * 10**9, 100 * 10**9): 350,  # 70b to 100b
            (100 * 10**9, 400 * 10**9): 400  # 100b to 400b
        }

        rps_scores = {
            (1, 2): 30,
            (2, 5): 70,
            (5, 10): 100,
            (10, 50): 120,
            (50, 100): 190,
            (100, 200): 220,
            (200, 500): 250,
            (500, 1000): 290,
            (1000, 2000): 350,
            (2000, 3000): 500,
        }
        
        latency_scores = {
            (20000, float('inf')): 30,
            (10000, 20000): 60,
            (7000, 10000): 80,
            (5000, 7000): 100,
            (2000, 5000): 120,
            (1000, 2000): 140,
            (500, 1000): 160,
            (100, 500): 280,
            (0, 100): 300
        }

        gb_tb_scores = {
            (1, 100): 20,          
            (100, 500): 50,          
            (500, 1000): 80,         
            (1000, 5000): 100,       
            (5000, 20000): 120,       
        }

        data_type_scores = {
            "word": 20,
            "excel": 40,
            "pdf" :60,
            "database": 80,
            "images": 100,
            "video": 120
        }

        param_value = self.get_parameter_value(model_parameters)
        latency_value = self.parse_latency(latency)
        data_size_value_parsed = self.parse_data_size(data_size_value)
        param_rank = next(score for (low, high), score in model_param_scores.items() if low <= param_value < high)
        rps_rank = next(score for (low, high), score in rps_scores.items() if low <= int(rps) < high)
        latency_rank = next(score for (low, high), score in latency_scores.items() if low <= latency_value < high)
        data_size_rank = next(score for (low, high), score in gb_tb_scores.items() if low <= data_size_value_parsed < high)
        data_type_rank = data_type_scores.get(data_type_value.lower(), 0)  
        total_score = param_rank + rps_rank + latency_rank + data_size_rank + data_type_rank
        print(f"### Calculated Total Score: {total_score} (Model Parameter Rank: {param_rank}, \
              RPS Rank: {rps_rank}, Latency Rank: {latency_rank}, Data Rank: {data_size_rank}, Data Type Rank: {data_type_rank})")
        
        return self.calc_generalized_pod_requirements( total_score)

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
        model_name_field = next((field for field in state["fields"] if field["name"] == "model_name"), None)
        model_parameters_field = next((field for field in state["fields"] if field["name"] == "model_parameters"), None)
        rps_field = next((field for field in state["fields"] if field["name"] == "rps"), None)
        latency_field = next((field for field in state["fields"] if field["name"] == "latency"), None)
        data_size_field = next((field for field in state["fields"] if field["name"] == "data_size"), None)
        data_type_field = next((field for field in state["fields"] if field["name"] == "data_type"), None)
        model_name_value = model_name_field["value"] if model_name_field else None
        model_parameters_value = model_parameters_field["value"] if model_parameters_field else None
        rps_value = rps_field["value"] if rps_field else None
        latency_value = latency_field["value"] if latency_field else None
        data_size_value = data_size_field["value"] if latency_field else None
        data_type_value = data_type_field["value"] if latency_field else None
        hw_sizing_pod = self.selectHardwareSizingPod(model_name_value, model_parameters_value, rps_value, latency_value, data_size_value, data_type_value)
        compute_nodes_required = hw_sizing_pod.get("compute_nodes_required")
        pcie_nodes_required = hw_sizing_pod.get("pcie_nodes_required")
        gpus_required = hw_sizing_pod.get("gpus_required")

        prompt = (
            f"Given the '{content}', let the user know that AI infrastructure sizing recommendations are "
            f"{compute_nodes_required} compute nodes, {pcie_nodes_required} PCIe nodes, and {gpus_required} GPUs. "
            f"Respond exactly with the phrase 'Here's the recommended AI Infrastructure Hardware Sizing' at the start of the response, followed by the information in this format:\n"
            f"Model: {model_name_value}\n"
            f"Model Parameters: {model_parameters_value}\n"
            f"RPS: {rps_value}\n"
            f"Latency: {latency_value}\n"
            f"Data Type: {data_type_value}\n"
            f"Data Size: {data_size_value}\n\n"
            

            f"Recommended configuration:\n"
            f"Compute Nodes: {compute_nodes_required}\n"
            f"PCIe Nodes: {pcie_nodes_required}\n"
            f"GPUs: {gpus_required}\n"
        )

        finalsummary_response = self.model.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192" 
        )

        gpu_recommendation = finalsummary_response.choices[0].message.content.strip()

        
        print("######## gpu_recommendation: ", gpu_recommendation)


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
    outMsg = []
    done = False
    for msg in result['messages']:
        if isinstance(msg, HumanMessage):
            inputs.append(msg)
    for msg in result['messages']:
        if isinstance(msg, ToolMessage):
            if msg.content == 'done':
                done = True
                break
            outMsg.append(msg.content)
    if done:
        outMsg = []
        for event in abot.graph.stream(None, thread, stream_mode="values"):
            msg_content = event["messages"][-1].content
            if msg_content not in outMsg and msg_content != "done":
                outMsg.append(msg_content)
    return outMsg[0], done

def getResponse(user_input, json_data):
    try:
        try:
            json_data = json.loads(json_data)
        except ValueError:
            trace_back = traceback.format_exc()
            logger.error("getResponse " + str(trace_back))
        # json_data = json.loads(json_data)
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        if isinstance(json_data, dict):
            json_data["messages"] = [HumanMessage(content=input) for input in user_input]
        else:
            logger.error("getResponse: `json_data` is not a dictionary after parsing.")
            return
        json_data["messages"] = [HumanMessage(content=input) for input in user_input]
        return process(json_data)
    except Exception :
        print(f"error in thius is:- \n {traceback.format_exc()}")

json_data_hwsize = """{
              
                "system_prompt": "You are an AI infrastructure sizing assistant. Help users determine their hardware requirements based on their AI model needs.",\
                "default_prompt": "Please provide the following details about your AI model deployment:\
            - Model name\
                - Model parameters (in billions, e.g., '13b')\
                - Required requests per second (RPS)\
                - Target latency (in milliseconds)\
                - Data size (in GB or TB)\
                - Data type (word, excel, pdf, database, images, or video)",\
                "final_prompt": "Based on your requirements, I'll recommend the appropriate hardware configuration.",\
                "fields": [
            {
                "name": "model_name",
                "description": "Name of the AI model being deployed",
                "value": null,
                "default": "llama2",
                "prompt": "What is the name of the AI model you want to deploy?"
            },
            {
                "name": "model_parameters",
                "description": "Number of parameters in the model (in billions)",
                "value": null,
                "default": "7b",
                "prompt": "How many parameters does your model have? (Please specify in billions, e.g., '13b')"
            },
            {
                "name": "rps",
                "description": "Required requests per second",
                "value": null,
                "default": "10",
                "prompt": "How many requests per second (RPS) do you need to handle?"
            },
            {
                "name": "latency",
                "description": "Target latency in milliseconds",
                "value": null,
                "default": "100ms",
                "prompt": "What is your target latency requirement? (in milliseconds, e.g., '100ms')"
            },
            {
                "name": "data_size",
                "description": "Size of the data to be processed",
                "value": null,
                "default": "100GB",
                "prompt": "What is the size of your data? (Please specify in GB or TB)"
            },
            {
                "name": "data_type",
                "description": "Type of data being processed (word, excel, pdf, database, images, or video)",
                "value": null,
                "default": "database",
                "prompt": "What type of data will you be processing? (word, excel, pdf, database, images, or video)"
            }
        ],
        "messages": []
    }
"""

def agenticWorkflowAISizingTools(prompt_string, app_config):
    logger.info("Coming inside agenticWorkflowAISizingTools...")
    response, reset = getResponse(prompt_string, app_config["json_data"])
    if isinstance(response, str):
        cleaned_response = response.strip('"\'')
        return cleaned_response, reset
    else:
        return str(response), reset


if __name__ == "__main__":
    prompt_string = ["I want to deploy a llama2 model with 13b parameters"]
    app_config = {
    "json_data": json.dumps(json_data_hwsize)}
    result, reset = agenticWorkflowAISizingTools(prompt_string, app_config)
    print(f'Result : {result}\nReset : {reset}')
    if reset:
        string = input(f"{result}: ")
        prompt_string.append(string)
        result, reset = agenticWorkflowAISizingTools(prompt_string, app_config)
        print(f"result:- {result}")
    else:
        print(result)