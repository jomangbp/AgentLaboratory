import tiktoken

class TokenProcessor:
    def __init__(self, max_tokens=5000, chunk_size=4000, overlap_size=500, avg_chars_per_token=4):
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.avg_chars_per_token = avg_chars_per_token
        
    def clip_tokens(self, messages, model="gpt-4", max_tokens=100000):
        if str(model).lower().startswith("groq-"):
            max_tokens = 6000  # Límite fijo para Groq
            enc = tiktoken.encoding_for_model("gpt-4")  # Usar tokenizador base
        else:
            enc = tiktoken.encoding_for_model(model)
            
        total_tokens = sum([len(enc.encode(message["content"])) for message in messages])

        if total_tokens <= max_tokens:
            return messages  # No need to clip if under the limit

        tokenized_messages = []
        for message in messages:
            tokenized_content = enc.encode(message["content"])
            tokenized_messages.append({"role": message["role"], "content": tokenized_content})

        all_tokens = [token for message in tokenized_messages for token in message["content"]]
        clipped_tokens = all_tokens[total_tokens - max_tokens:]

        clipped_messages = []
        current_idx = 0
        for message in tokenized_messages:
            message_token_count = len(message["content"])
            if current_idx + message_token_count > len(clipped_tokens):
                clipped_message_content = clipped_tokens[current_idx:]
                clipped_message = enc.decode(clipped_message_content)
                clipped_messages.append({"role": message["role"], "content": clipped_message})
                break
            else:
                clipped_message_content = clipped_tokens[current_idx:current_idx + message_token_count]
                clipped_message = enc.decode(clipped_message_content)
                clipped_messages.append({"role": message["role"], "content": clipped_message})
                current_idx += message_token_count
        return clipped_messages

    def estimate_tokens(self, text):
        """Estima el número aproximado de tokens en un texto"""
        if not text:
            return 0
        return len(text) // self.avg_chars_per_token

    def truncate_text(self, text, max_length):
        """Helper para truncar texto"""
        if text and len(text) > max_length:
            return text[:max_length] + "... [truncated]"
        return text

    def split_into_chunks(self, text):
        """Helper para dividir texto en chunks con overlap"""
        if text is None:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            if end < text_len:
                while end > start + self.chunk_size - self.overlap_size and text[end-1] != ' ':
                    end -= 1
                chunk = text[start:end]
                if self.estimate_tokens(chunk) > self.chunk_size:
                    end = start + (self.chunk_size * self.avg_chars_per_token)
            else:
                end = text_len
            
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap_size if end < text_len else text_len
            
        return chunks

    def process_response(self, response, model):
        """Procesa la respuesta dividiéndola en chunks si es necesario"""
        if not response or not isinstance(response, str):
            return response
            
        if str(model).lower().startswith("groq-") or self.estimate_tokens(response) <= self.max_tokens:
            message = [{"role": "assistant", "content": response}]
            message = self.clip_tokens(message, model=model, max_tokens=self.max_tokens)
            return message[0]["content"]
            
        chunks = self.split_into_chunks(response)
        processed_chunks = []
        
        for chunk in chunks:
            message = [{"role": "assistant", "content": chunk}]
            message = self.clip_tokens(message, model=model, max_tokens=self.max_tokens)
            processed_chunks.append(message[0]["content"])
        
        return "\n".join(processed_chunks)

# Create global instance
token_processor = TokenProcessor()

# Export the functions that other modules need
def clip_tokens(messages, model="gpt-4", max_tokens=100000):
    return token_processor.clip_tokens(messages, model, max_tokens)

def process_response(response, model):
    return token_processor.process_response(response, model)