# !pip install --upgrade pyautogen
# !pip install python_dotenv

from autogen import ConversableAgent
from dotenv import load_dotenv
import os
import pprint

# Load environment variables
load_dotenv('.secrets')
api_key = os.getenv("OPENAI_API_KEY")

# Correct LLM configuration structure
openAI_config = {
    "config_list": [
        {
            "model": "gpt-3.5-turbo",
            "api_key": api_key,
        }
    ],
    "timeout": 120  # Recommended to add timeout
}

llama_config = {
    "config_list": [
        {
            "model": "meta-llama-3.1-8b-instruct",
            "base_url": "http://192.168.1.170:1234/v1",
            "api_key": "lm-studio",
        }
    ],
    "timeout": 120  # Recommended to add timeout
}

# Create agents with validated configuration
bret = ConversableAgent(
    name="MattDamon",
    llm_config=openAI_config,
    system_message="""You are the real life agent, whose life was portrayed in movies like 'Wolf of Wall Street'/ 'War Dogs'. 
                        You want to invest your last $100 in stock market.  Something that you can buy today and sell tomorrow, to keep the extra cash.
                        Keep responses under 2 sentences.
                        When you're ready to end the conversation say 'I gotta go'.""",
    human_input_mode= "NEVER",
    is_termination_msg= lambda msg:" I gotta go" in msg["content"],
)

jemaine = ConversableAgent(
    name = "WarrenBuffet",
    llm_config = llama_config,
    system_message = """You are the world famous (Warren Buffet + Satoshi Nakamoto) combined. 
                        You use all your knowledge to invest money into a stock today, and generate the maximum profit tomorrow.
                        Keep responses under 2 sentences.
                        When you're ready to end the conversation say 'I gotta go'.""",
    human_input_mode = "NEVER",
    is_termination_msg = lambda msg:"I gotta go" in msg["content"],

)

# Initiate chat with error handling
try:
    chat_result = bret.initiate_chat(
        recipient=jemaine,
        message="Jemaine, explain recursion using a pizza analogy!",
        max_turns=2
    )
except Exception as e:
    print(f"Chat failed: {str(e)}")
    exit(1)

# Display results
print("\n=== Conversation ===")
pprint.pprint(chat_result.chat_history)

print("\n=== Summary ===")
pprint.pprint(chat_result.summary)

print("\n=== Cost ===")
pprint.pprint(chat_result.cost)