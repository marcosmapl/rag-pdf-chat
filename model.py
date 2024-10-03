from langchain.llms import BaseLLM

class LLMWrapper(BaseLLM):
    def __init__(self, model):
        self.model = model

    def _call(self, prompt: str, **kwargs):
        response = self.model.generate(prompt)
        return response["text"]
