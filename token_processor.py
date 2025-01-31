import tiktoken
from functools import lru_cache
from typing import List, Dict, Union, Optional
import concurrent.futures

class TokenProcessor:
    def __init__(self, max_tokens=5000, chunk_size=4000, overlap_size=500, avg_chars_per_token=4):
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.avg_chars_per_token = avg_chars_per_token
        self._encoders = {}
        self._model_limits = {
            "groq-": 6000,
            "ollama-": 16384,
            "lmstudio-": 16384,
            "gpt-4": 8192,
            "gpt-3.5": 4096,
            "default": 4096
        }
        
    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        if model not in self._encoders:
            try:
                if str(model).lower().startswith(("groq-", "ollama-", "lmstudio-")):
                    self._encoders[model] = tiktoken.get_encoding("cl100k_base")
                else:
                    self._encoders[model] = tiktoken.encoding_for_model(model)
            except (KeyError, ValueError):
                print(f"Warning: Using default tokenizer for unknown model: {model}")
                self._encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self._encoders[model]

    def _get_model_limit(self, model: str) -> int:
        model_lower = str(model).lower()
        for prefix, limit in self._model_limits.items():
            if model_lower.startswith(prefix):
                return limit
        return self._model_limits["default"]

    @lru_cache(maxsize=1024)
    def _cached_encode(self, text: str, model: str) -> List[int]:
        try:
            if not text:
                return []
            if len(text) > 100000:
                parts = []
                for i in range(0, len(text), 100000):
                    part = text[i:i + 100000]
                    parts.extend(self._get_encoder(model).encode(part))
                return parts
            return self._get_encoder(model).encode(text)
        except Exception as e:
            print(f"Warning: Encoding error for model {model}: {e}")
            return []

    def clip_tokens(self, messages: List[Dict[str, str]], model: str = "gpt-4", max_tokens: int = None) -> List[Dict[str, str]]:
        if max_tokens is None:
            max_tokens = self._get_model_limit(model)
            
        enc = self._get_encoder(model)
        try:
            total_tokens = sum(len(self._cached_encode(message["content"], model)) for message in messages)
        except Exception as e:
            print(f"Warning: Error encoding messages: {e}")
            return messages

        if total_tokens <= max_tokens:
            return messages

        tokenized_messages = []
        for message in messages:
            tokenized_content = self._cached_encode(message["content"], model)
            tokenized_messages.append({"role": message["role"], "content": tokenized_content})

        all_tokens = [token for message in tokenized_messages for token in message["content"]]
        clipped_tokens = all_tokens[-max_tokens:]

        clipped_messages = []
        current_idx = 0
        for message in tokenized_messages:
            message_token_count = len(message["content"])
            if current_idx + message_token_count > len(clipped_tokens):
                clipped_message_content = clipped_tokens[current_idx:]
                clipped_message = enc.decode(clipped_message_content)
                if clipped_message.strip():
                    clipped_messages.append({"role": message["role"], "content": clipped_message})
                break
            else:
                clipped_message_content = clipped_tokens[current_idx:current_idx + message_token_count]
                clipped_message = enc.decode(clipped_message_content)
                if clipped_message.strip():
                    clipped_messages.append({"role": message["role"], "content": clipped_message})
                current_idx += message_token_count
        return clipped_messages

    def split_into_chunks(self, text: str) -> List[str]:
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        dynamic_chunk_size = min(self.chunk_size, max(1000, text_len // 10))
        dynamic_overlap = min(self.overlap_size, dynamic_chunk_size // 4)
        
        while start < text_len:
            end = start + dynamic_chunk_size
            if end < text_len:
                sentence_markers = ['. ', '? ', '! ', '.\n', '?\n', '!\n']
                found_end = False
                
                for marker in sentence_markers:
                    sentence_end = text.rfind(marker, start, end)
                    if sentence_end > start:
                        end = sentence_end + len(marker)
                        found_end = True
                        break
                
                if not found_end:
                    while end > start + dynamic_chunk_size - dynamic_overlap and text[end-1] != ' ':
                        end -= 1
            else:
                end = text_len
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - dynamic_overlap if end < text_len else text_len
        
        return chunks

    def process_response(self, response: Union[str, None], model: str) -> Union[str, None]:
        if not response or not isinstance(response, str):
            return response
            
        try:
            if '<think>' in response:
                chunks = self.split_into_chunks(response)
                if chunks:
                    return chunks[0]
                return response
            
            model_lower = str(model).lower()
            max_tokens = self._get_model_limit(model_lower)
            
            if model_lower.startswith(("groq-", "lmstudio-", "ollama-")):
                self.chunk_size = min(4000, max_tokens // 2)
                self.overlap_size = self.chunk_size // 3
            
            if self.estimate_tokens(response) <= max_tokens:
                return response
            
            chunks = self.split_into_chunks(response)
            if not chunks:
                return None
                
            return self._process_continuous_response(chunks, model)
        except Exception as e:
            print(f"Warning: Error processing response: {e}")
            return response

    def _process_continuous_response(self, chunks: List[str], model: str) -> str:
        if not chunks:
            return ""
        
        if len(chunks) == 1:
            return chunks[0]
        
        result = chunks[0]
        
        for chunk in chunks[1:]:
            if not chunk.strip():
                continue
            
            if not result.endswith(('...', '…', '.', '!', '?')):
                result += ' '
            result += chunk.strip()
        
        return result

    def batch_process_messages(self, messages: List[Dict[str, str]], model: str) -> List[Dict[str, str]]:
        processed_messages = []
        for message in messages:
            if "content" in message and message["content"]:
                processed_content = self.process_response(message["content"], model)
                if processed_content:
                    processed_messages.append({**message, "content": processed_content})
            else:
                processed_messages.append(message)
        return processed_messages

    def _process_chunk(self, chunk: str, model: str) -> str:
        if not chunk:
            return ""
        message = [{"role": "assistant", "content": chunk}]
        message = self.clip_tokens(message, model=model, max_tokens=self._get_model_limit(model))
        return message[0]["content"]

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text) // self.avg_chars_per_token

    def truncate_text(self, text: str, max_length: int) -> str:
        if text and len(text) > max_length:
            return text[:max_length] + "... [truncated]"
        return text

token_processor: TokenProcessor = TokenProcessor()

__all__ = ['token_processor', 'clip_tokens', 'process_response']

clip_tokens = token_processor.clip_tokens
process_response = token_processor.process_response
