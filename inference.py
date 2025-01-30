import time, tiktoken
from openai import OpenAI
import openai
import os, anthropic, json
from token_processor import clip_tokens, process_response


TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 0.14 / 1000000,      # $0.14 per 1M tokens (cache miss)
        "deepseek-reasoner": 0.55 / 1000000,  # $0.55 per 1M tokens (cache miss)
        "o1": 15.00 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00 / 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 0.28 / 1000000,      # $0.28 per 1M tokens
        "deepseek-reasoner": 2.19 / 1000000,  # $2.19 per 1M tokens
        "o1": 60.00 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

import requests

def query_model(model_str, prompt, system_prompt, openai_api_key=None, anthropic_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    preloaded_api = os.getenv('OPENAI_API_KEY')
    deepseek_api = os.getenv('DEEPSEEK_API_KEY')
    
    # Handle LM Studio models
    if model_str.startswith("lmstudio-"):
        for _ in range(tries):
            try:
                client = OpenAI(
                    base_url="http://localhost:1234/v1",
                    api_key="lm-studio"
                )
                model_name = model_str.replace('lmstudio-', '')
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temp if temp is not None else 0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LM Studio API Error: {e}")
                time.sleep(timeout)
                continue
        raise Exception("Max retries: timeout for LM Studio")

    # Add Ollama model handling
    if model_str.startswith("ollama-"):
        for _ in range(tries):
            try:
                ollama_model = model_str.replace("ollama-", "")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": ollama_model,
                        "messages": messages,
                        "stream": False,
                        "temperature": temp if temp is not None else 0.7
                    }
                )
                response.raise_for_status()
                return response.json()["message"]["content"]
            except Exception as e:
                print(f"Ollama API Error: {e}")
                time.sleep(timeout)
                continue
        raise Exception("Max retries: timeout for Ollama")

    # Check for DeepSeek models first
    if model_str in ["deepseek-chat", "deepseek-reasoner"]:
        if not deepseek_api:
            raise Exception("No DeepSeek API key provided")
    elif openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    elif openai_api_key is None and anthropic_api_key is None:
        raise Exception("No API key provided in query_model function")

    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "deepseek-chat" or model_str == "deepseek-reasoner":
                model_str = model_str
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model=model_str,
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model=model_str,
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content

            if model_str in ["o1-preview", "o1-mini", "claude-3-5-sonnet", "o1"]:
                encoding = tiktoken.encoding_for_model("gpt-4o")
            elif model_str in ["deepseek-chat", "deepseek-reasoner"]:
                encoding = tiktoken.encoding_for_model("cl100k_base")
            else:
                encoding = tiktoken.encoding_for_model(model_str)
            if model_str not in TOKENS_IN:
                TOKENS_IN[model_str] = 0
                TOKENS_OUT[model_str] = 0
            TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
            TOKENS_OUT[model_str] += len(encoding.encode(answer))
            if print_cost:
                print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))

# Test at the bottom of the file
#print(query_model(
   #model_str="ollama-deepseek-r1:1.5b", 
   #prompt="What is machine learning?", 
   #system_prompt="You are a helpful AI assistant."
#))
