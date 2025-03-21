# filename: planningStockReport.py
import os
# PDB
import pdb
# Date time for coding task
import datetime
# Pretty print
import autogen
from autogen import ConversableAgent, AssistantAgent, initiate_chats
# Command Line Executor
from autogen.code_utils import create_virtual_env
from autogen.coding import CodeBlock, LocalCommandLineCodeExecutor
# OpenAI for both chatGPT and Deepseek
from openai import OpenAI
# Get environment variables
from dotenv import load_dotenv
import pprint

load_dotenv('.secrets')
openAI_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

openAI_config = {
    "config_list": [
        {
            "model": "gpt-3.5-turbo",
            "api_key": openAI_api_key,
        }
    ],
    "timeout": 120  
}

llama_config = {
    "config_list": [
        {
            "model": "meta-llama-3.1-8b-instruct",
            "base_url": "http://192.168.1.170:1234/v1",
            "api_key": "lm-studio",
        }
    ],
    "timeout": 120  
}

deepseek_coder_config = {
    "config_list": [
        {
            "model": "deepseek-coder-v2-lite-instruct-mlx",
            "base_url": "http://192.168.1.170:1234/v1",
            "api_key": "deepseek-coder-v2-lite-instruct-mlx",
        }
    ],
    "timeout": 120  
}

deepseek_reasoner_config = {
    "config_list": [
        {
            "model": "deepseek-reasoner",
            "price": [0.00014, 0.00219],  # [prompt_price_per_1k, completion_token_price_per_1k]
            "base_url": "https://api.deepseek.com",
            "api_key": deepseek_api_key,
        }
    ],
    "timeout": 120  
}

# ======================================
user_proxy = ConversableAgent(
    name="Admin",
    system_message="Give the task, and send instructions to writer to refine the blog post.",
    code_execution_config=False,
    llm_config=deepseek_coder_config,
    human_input_mode="ALWAYS", # If user does not provide feedback, the LLM will do it for us
)

planner = ConversableAgent(
    name="Planner",
    system_message="Given a task, please determine what information is needed to complete the task. Please note that the information will all be retrieved using Python code. Please only suggest information that can be retrieved using Python code.",
    description="Planner. Given a task, determine what information is needed to complete the task. After each step is done by others, check the progress and instruct the remaining steps.",
    llm_config=deepseek_coder_config,
)

# system msg: Already has a default one because it is a code writer
engineer = AssistantAgent(
    name="Engineer",
    llm_config=deepseek_coder_config,
    description="An engineer that writes code based on the plan provided by the planner.",
)

# Check engineer system prompt message
print(engineer.system_message)

executor = ConversableAgent(
    name="Executor",
    system_message="Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False,
    },
)

writer = ConversableAgent(
    name="Writer",
    llm_config=deepseek_coder_config,
    system_message="""Writer.
Please write blogs in markdown format (with relevant titles) and put the content in pseudo ```md``` code block. You take feedback from the admin and refine your blog.""",
    description="Writer."
    "Write blogs based on the code execution results and take feedback from the admin to refine the blog."
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, engineer, writer, executor, planner],
    messages=[], # No initial message
    max_round=20,
)

manager = autogen.GroupChatManager(
    groupchat=groupchat, 
    llm_config=deepseek_coder_config
)

groupchat_result = user_proxy.initiate_chat(
    manager, # We initiate between user and manager so we can give the task
    message="Write a blogpost about the stock price performance of Nvidia in the past month. Today's date is 2024-07-26.",
)
