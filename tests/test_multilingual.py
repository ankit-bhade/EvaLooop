"""Unit tests for the multilingual summary loop."""

import pytest

from evaloop.core.config import EvaluationConfig
from evaloop.tasks import (
    CodeSummarizationTask,
    MultilingualCodeSummarizationTask,
    TaskFactory,
)
from evaloop.tasks.code_summarization_multilingual import STARTER_PHRASES

from conftest import ASSERT_LINE, MBPP_STYLE_PROMPT

SPANISH_CONFIG = {"type": "code_summarization_multilingual", "summary_language": "spanish"}


class TestMultilingualCodeSummarizationTask:
    def test_create_prompt_names_language_and_starter(self):
        task = MultilingualCodeSummarizationTask(SPANISH_CONFIG)
        prompt = task.create_prompt("def f(x): return x")
        assert "def f(x): return x" in prompt
        assert "written in Spanish" in prompt
        assert prompt.count(STARTER_PHRASES["spanish"]) == 2
        assert "write a python function to" not in prompt

    def test_default_prompt_mirrors_standard_prompt(self, summarization_config):
        # Only the language and the opening phrase may differ from the standard loop
        english = CodeSummarizationTask(summarization_config).create_prompt("CODE")
        spanish = MultilingualCodeSummarizationTask(SPANISH_CONFIG).create_prompt("CODE")
        assert spanish.replace(" written in Spanish", "").replace(
            STARTER_PHRASES["spanish"], "write a python function to"
        ) == english

    def test_language_is_case_insensitive(self):
        task = MultilingualCodeSummarizationTask({"summary_language": " Spanish "})
        assert task.summary_language == "spanish"

    def test_custom_starter_phrase(self):
        task = MultilingualCodeSummarizationTask(
            {"summary_language": "italian", "starter_phrase": "scrivi una funzione Python che"}
        )
        assert "scrivi una funzione Python che" in task.create_prompt("CODE")

    def test_unknown_language_without_starter_raises(self):
        with pytest.raises(ValueError, match="starter phrase"):
            MultilingualCodeSummarizationTask({"summary_language": "klingon"})

    def test_missing_language_raises(self):
        with pytest.raises(ValueError, match="summary_language"):
            MultilingualCodeSummarizationTask({})

    def test_postprocess_retains_assert_line(self):
        task = MultilingualCodeSummarizationTask(SPANISH_CONFIG)
        summary = "escribe una función de Python que encuentre los elementos comunes"
        result = task.postprocess(summary, {"cycle_input": MBPP_STYLE_PROMPT})
        assert result == f'"""\n{summary}\n{ASSERT_LINE}\n"""\n'

    def test_assert_line_survives_several_cycles(self):
        task = MultilingualCodeSummarizationTask(SPANISH_CONFIG)
        cycle_input = MBPP_STYLE_PROMPT
        for cycle in range(3):
            cycle_input = task.postprocess(f"resumen {cycle}", {"cycle_input": cycle_input})
        assert cycle_input.splitlines()[-2] == ASSERT_LINE

    def test_is_not_testable(self):
        assert MultilingualCodeSummarizationTask(SPANISH_CONFIG).produces_testable_code is False

    def test_registered_with_factory(self):
        task = TaskFactory().create_task(SPANISH_CONFIG)
        assert isinstance(task, MultilingualCodeSummarizationTask)


class TestSummaryLanguageConfig:
    def make_config(self, tmp_path, **kwargs):
        return EvaluationConfig(model="gpt-4o-mini", output_dir=str(tmp_path), **kwargs)

    def test_default_is_standard_loop(self, tmp_path, summarization_config):
        configs = self.make_config(tmp_path).get_task_configs()
        assert [c["type"] for c in configs] == ["code_generation", "code_summarization"]
        assert configs[1] == summarization_config

    @pytest.mark.parametrize("language", ["english", "English", ""])
    def test_english_means_standard_loop(self, tmp_path, language):
        config = self.make_config(tmp_path, summary_language=language)
        assert config.summary_language is None
        assert config.get_task_configs()[1]["type"] == "code_summarization"

    def test_other_language_selects_multilingual_task(self, tmp_path, generation_config):
        config = self.make_config(tmp_path, summary_language="Spanish")
        configs = config.get_task_configs()
        assert configs[0] == generation_config
        assert configs[1] == {
            "type": "code_summarization_multilingual",
            "summary_language": "spanish",
        }
        tasks = [TaskFactory().create_task(c) for c in configs]
        assert isinstance(tasks[1], MultilingualCodeSummarizationTask)

    def test_experiment_name_marks_language(self, tmp_path):
        assert "_summary_spanish_" in self.make_config(tmp_path, summary_language="spanish").experiment_name
        assert "_summary_" not in self.make_config(tmp_path).experiment_name
