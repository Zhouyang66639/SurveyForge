import concurrent.futures
import json
import os
import threading
import time

import anthropic
import requests
import torch
import transformers
from openai import OpenAI
from tqdm import tqdm

class APIModel:

    def __init__(self, model, api_key, api_url) -> None:
        self.__api_key = api_key
        self.__api_url = api_url
        self.model = model

    def _resolve_responses_url(self):
        url = self.__api_url.rstrip("/")
        if url.endswith("/responses"):
            return url
        if url.endswith("/chat/completions"):
            return f"{url[:-len('/chat/completions')]}/responses"
        if url.endswith("/v1"):
            return f"{url}/responses"
        return f"{url}/v1/responses"

    def _resolve_chat_url(self):
        url = self.__api_url.rstrip("/")
        if url.endswith("/responses"):
            return f"{url[:-len('/responses')]}/chat/completions"
        if url.endswith("/chat/completions"):
            return url
        if url.endswith("/v1"):
            return f"{url}/chat/completions"
        return f"{url}/v1/chat/completions"

    def _should_use_responses(self):
        wire_api = os.getenv("WIRE_API", "").strip().lower()
        if wire_api == "responses":
            return True
        if wire_api == "chat":
            return False

        model_name = self.model.lower()
        # gpt-5/codex family is commonly exposed via responses API on proxy providers.
        return ("gpt-5" in model_name) or ("codex" in model_name)

    def _parse_chat_response(self, response_json):
        choices = response_json.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        return None

    def _parse_responses_response(self, response_json):
        output_text = response_json.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        texts = []
        for output_item in response_json.get("output", []):
            if not isinstance(output_item, dict):
                continue
            for content_item in output_item.get("content", []):
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") in {"output_text", "text"}:
                    text_value = content_item.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        texts.append(text_value)
                    elif isinstance(text_value, dict):
                        nested = text_value.get("value")
                        if isinstance(nested, str) and nested.strip():
                            texts.append(nested)
        if texts:
            return "\n".join(texts)
        return None

    def _request_openai_compat(self, text, temperature, max_try):
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.__api_key}",
            "User-Agent": "SurveyForge/1.0",
            "Content-Type": "application/json",
        }

        payload_chat = {
            "model": f"{self.model}",
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": f"{text}",
                }
            ],
        }

        payload_responses = {
            "model": f"{self.model}",
            "input": f"{text}",
        }
        # Some providers reject temperature in responses mode; only send non-default value.
        if temperature != 1:
            payload_responses["temperature"] = temperature

        wire_api = os.getenv("WIRE_API", "").strip().lower()
        if wire_api == "responses":
            modes = ["responses"]
        elif wire_api == "chat":
            modes = ["chat"]
        else:
            prefer_responses = self._should_use_responses()
            modes = ["responses", "chat"] if prefer_responses else ["chat", "responses"]
        last_err = None

        for attempt in range(max_try):
            for mode in modes:
                try:
                    if mode == "responses":
                        url = self._resolve_responses_url()
                        payload = json.dumps(payload_responses)
                    else:
                        url = self._resolve_chat_url()
                        payload = json.dumps(payload_chat)

                    response = requests.request("POST", url, headers=headers, data=payload, timeout=180)
                    response_json = json.loads(response.text)

                    if response.status_code >= 400:
                        raise RuntimeError(f"{mode} HTTP {response.status_code}: {response_json}")

                    if mode == "responses":
                        parsed_text = self._parse_responses_response(response_json)
                    else:
                        parsed_text = self._parse_chat_response(response_json)

                    if parsed_text is not None:
                        return parsed_text

                    raise RuntimeError(f"{mode} invalid response: {response_json}")
                except Exception as req_err:
                    last_err = req_err
                    print(f"API request failed (mode={mode}, attempt {attempt + 1}/{max_try}): {req_err}")
            time.sleep(0.5)

        raise RuntimeError(f"API request failed after {max_try} attempts, last error: {last_err}")

    def __req(self, text, temperature, max_try = 10):
        if "deepseek" in self.model:
            for _ in range(max_try):
                try:
                    client = OpenAI(
                        api_key=self.__api_key,
                        base_url=self.__api_url,
                    )
                    completion = client.chat.completions.create(
                        model=self.model,  # https://help.aliyun.com/zh/model-studio/getting-started/models
                        messages=[
                            {'role': 'user', 'content': f'{text}'}
                            ]
                    )
                    return completion.choices[0].message.content
                except Exception as e:
                    print(f"错误信息：{e}\n Retrying...{_} Times")
                    continue
                    # print("Ref：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        elif "claude"  not in self.model:
            return self._request_openai_compat(text=text, temperature=temperature, max_try=max_try)
        else:
            try:
                client = anthropic.Anthropic(api_key=self.__api_key)
                message = client.messages.create(
                    model=self.model,
                    max_tokens=4096, 
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": text}
                    ]
                )
                return message.content[0].text
            except Exception as e:
                print(f"Error occurred with Claude API: {e}")
                return None
    
    def chat(self, text, temperature=1):
        response = self.__req(text, temperature=temperature, max_try=5)
        return response

    def __chat(self, text, temperature, res_l, idx):
        
        response = self.__req(text, temperature=temperature)
        res_l[idx] = response
        return response

    def batch_chat(self, text_batch, temperature=0):
        max_threads = 100  # limit max concurrent threads using model API
        try:
            max_threads = max(1, int(os.getenv("SF_MAX_API_THREADS", "100")))
        except ValueError:
            max_threads = 100
        res_l = ['No response'] * len(text_batch)

        def chat_wrapper(text, temp, res_list, idx):
            self.__chat(text, temp, res_list, idx)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = []
            for i, text in enumerate(text_batch):
                future = executor.submit(chat_wrapper, text, temperature, res_l, i)
                futures.append(future)
                
                while len(futures) >= max_threads:
                    done, not_done = concurrent.futures.wait(futures, timeout=60, return_when=concurrent.futures.FIRST_COMPLETED)
                    futures = list(not_done)
                    time.sleep(10)  # Short delay to avoid busy-waiting

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing remaining"):
                future.result()

        return res_l

class LocalModel:

    def __init__(self, ckpt) -> None:
        self.ckpt = ckpt
        
        self._init_client()
        
    def _init_client(self):
        model_name =  self.ckpt.split('/')[-1]
        self.client = transformers.pipeline('text-generation',
                                             model=self.ckpt, 
                                             #model_kwargs={"torch_dtype": torch.bfloat16},
                                             device_map="auto")
        print(f"Model {model_name} loaded successfully")

    def _req(self, text, temperature, max_try = 5):
        message = [{"role": "user", "content": text}]
        response = self.client(message,
                            max_new_tokens=4096,
                            temperature=temperature,
                            pad_token_id=self.client.tokenizer.eos_token_id)
        return response[0]['generated_text'][-1]['content']
        # try:
        #     response = self.client(message,
        #                         max_new_tokens=256,
        #                         temperature=temperature)
        #     return response[0]['generated_text'][-1]['content']
        # except:
        #     for _ in range(max_try):
        #         try:
        #             response = self.client(message,
        #                                 max_new_tokens=256,
        #                                 temperature=temperature)
        #             return response[0]['generated_text'][-1]['content']
        #         except:
        #             pass
        #         time.sleep(0.2)
        #     return None

    def chat(self, text, temperature=1.0):
        response = self._req(text, temperature=temperature, max_try=5)
        return response
    
    def _batch_chat_i(self, text, temperature, res_l, idx):
        response = self._req(text, temperature=temperature)
        res_l[idx] = response
        return response
        
    def batch_chat(self, text_batch, temperature=1.0):
        max_threads=1
        res_l = ['No response'] * len(text_batch)
        thread_l = []
        for i, text in zip(range(len(text_batch)), text_batch):
            thread = threading.Thread(target=self._batch_chat_i, args=(text, temperature, res_l, i))
            thread_l.append(thread)
            thread.start()
            while len(thread_l) >= max_threads: 
                for t in thread_l:
                    if not t .is_alive():
                        thread_l.remove(t)
                time.sleep(0.3)
        
        for thread in tqdm(thread_l):
            thread.join()
        return res_l
