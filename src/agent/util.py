from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompt_values import ChatPromptValue

from config import settings


class InputTokenCounter:
    """A counter for the input tokens consumed by the used Large Language Model (LLM).

    This counter can be used in any LangChain chain involving a LangChain model to count
    the input tokens the model consumes over consecutive model invocations.

    Attributes:
        llm: Large Language Model whose consumed input tokens should be counted
        num_tokens: Number of input tokens consumed by the LLM so far
        last_count: Number of input tokens consumed by the last LLM call
    """

    def __init__(self, llm: BaseLanguageModel) -> None:
        self.llm = llm
        self.num_tokens = 0
        self.last_count = 0

    def count(self, prompt: ChatPromptValue) -> ChatPromptValue:
        """Count the number of input tokens the model will consume for the given prompt.

        Args:
            prompt: Chat prompt

        Returns:
            The given chat prompt
        """
        if settings.llm.count_input_tokens:
            # TODO: Using get_num_tokens_from_messages with our custom LangChain model wrapping the AI Core
            #  orchestration service results in the warning "Token indices sequence length is longer than the
            #  specified maximum sequence length for this model (2841 > 1024). Running this sequence through
            #  the model will result in indexing errors".
            #  The default implementation in the LangChain BaseLanguageModel class uses a GPT2Tokenizer to
            #  tokenize the messages to count the tokens. We should find a better way to count the tokens.
            input_tokens_for_prompt = self.llm.get_num_tokens_from_messages(prompt.to_messages())
            self.num_tokens += input_tokens_for_prompt
            self.last_count = input_tokens_for_prompt

        return prompt


class IdGenerator:
    """A very simple generator for fresh numeric IDs.

    This class is not intended for a multithreaded environment, i.e., there is no locking in place.
    As name generation is currently happening within a single agent session, this is by design.
    """

    def __init__(self, first_id: int = 0) -> None:
        """Create a new generator for numeric IDs.

        Args:
             first_id: First ID to be provided by the generator.
        """
        self._counter = first_id

    def fresh_id(self) -> int:
        """Generate a fresh numeric ID.

        Returns:
            The fresh numeric ID which has not been issued before.
        """
        fresh_id = self._counter
        self._counter += 1

        return fresh_id
