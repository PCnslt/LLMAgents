import os
# PDB
import pdb
# Date time for coding task
import datetime
# Pretty print

import autogen
# Agent 2: Code Executor agent
from autogen import ConversableAgent
from autogen import AssistantAgent
from autogen import initiate_chats
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

#  ======================================

venv_dir = ".env_llm"
venv_context = create_virtual_env(venv_dir)

executor = LocalCommandLineCodeExecutor(
    virtual_env_context=venv_context,
    timeout=200,
    work_dir="coding",
)
print(
    executor.execute_code_blocks(code_blocks=[CodeBlock(language="python", code="import sys; print(sys.executable)")])
)

# ---

# Agent that writes code
code_writer_agent = AssistantAgent(
    name="code_writer_agent",
    llm_config=deepseek_reasoner_config,
    code_execution_config=False,
    human_input_mode="NEVER",
)

# Code writer agent default prompt
code_writer_agent_system_message = code_writer_agent.system_message
print(code_writer_agent_system_message)


pdb.set_trace()


# Agent that executes code
code_executor_agent = ConversableAgent(
    name="code_executor_agent",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
    default_auto_reply=
    "Please continue. If everything is done, reply 'TERMINATE'.",
)

# ---


today = datetime.datetime.now().date()

message = f"Today is {today}. "\
"Create a plot showing the normalized price of NVDA and BTC-USD for the last 5 years "\
"with their 60 weeks moving average. "\
"Make sure the code  in markdown code block, print the normalized prices, save the figure"\
" to a file asset_analysis.png and show it. Provide all the code necessary in a single python bloc. "\
"Re-provide the code blisock that needs to be executed with each of your messages. "\
"If python packages are necessary to execute the code, provide a markdown "\
"sh block with only the command necessary to install them and no comments."

# Let's define the chat and initiate it !
chat_result = code_executor_agent.initiate_chat(
    code_writer_agent,
    message=message
)