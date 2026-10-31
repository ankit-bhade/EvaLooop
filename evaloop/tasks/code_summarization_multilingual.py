"""Code summarization in a natural language other than English.

This task powers the "multilingual summary" loop: generation still receives an
English instruction, but the description it implements is the model's own
summary written in another language. Everything else matches the standard
loop, so the only perturbation is the model's own non-English wording:

  - The prompt is the standard one-sentence summarization prompt with the
    target language added and the fixed opening phrase translated.
  - Output extraction is unchanged (the full stripped response).
  - The assert line from the cycle's input is re-attached by the inherited
    CodeSummarizationTask.postprocess, which keeps the function name and
    signature stable across cycles exactly as in the standard loop.
"""

from typing import Any, Dict

from .code_summarization import CodeSummarizationTask
from .factory import register_task

# Translations of the standard loop's fixed opening phrase
# "write a python function to". Add a language here, or pass
# "starter_phrase" in the task config, to use another one.
STARTER_PHRASES = {
    "spanish": "escribe una función de Python que",
    "french": "écris une fonction Python qui",
    "german": "schreibe eine Python-Funktion, die",
    "chinese": "编写一个 Python 函数，用于",
    "hindi": "एक Python फ़ंक्शन लिखिए जो",
}


@register_task("code_summarization_multilingual")
class MultilingualCodeSummarizationTask(CodeSummarizationTask):
    """Summarizes code in one sentence written in a non-English language."""

    def __init__(self, task_config: Dict[str, Any]):
        """Initialize task with configuration.

        Config keys:
          - "summary_language" (required): e.g. "spanish".
          - "starter_phrase" (optional): opening phrase in that language;
            defaults to STARTER_PHRASES[summary_language].
          - "prompt_template" (optional): must contain {code}; may use
            {language} and {starter_phrase}.
        """
        super().__init__(task_config)
        self.summary_language = task_config.get("summary_language", "").strip().lower()
        if not self.summary_language:
            raise ValueError("summary_language is required for code_summarization_multilingual")

        self.starter_phrase = task_config.get("starter_phrase") or STARTER_PHRASES.get(self.summary_language)
        if not self.starter_phrase:
            raise ValueError(
                f"No starter phrase known for '{self.summary_language}'. "
                f"Pass 'starter_phrase' or use one of: {', '.join(sorted(STARTER_PHRASES))}"
            )

    def create_prompt(self, input_text: str) -> str:
        """Create the summarization prompt, asking for the target language."""
        prompt_template = self.task_config.get("prompt_template",
            "Use one sentence written in {language} to summarize the following code "
            "and start with '{starter_phrase}':\n\n```\n{code}\n```\n\n```\n{starter_phrase}\n```"
        )
        return prompt_template.format(
            code=input_text,
            language=self.summary_language.capitalize(),
            starter_phrase=self.starter_phrase,
        )
