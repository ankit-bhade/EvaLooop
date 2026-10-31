"""Unit tests for endpoint models, task subsets and checkpoints (no network calls)."""

import json

import pytest

from evaloop.core.config import EvaluationConfig
from evaloop.core.run_state import load_checkpoint, save_checkpoint, select_task_subset
from evaloop.evaluation.cycle import BatchEvaluationCycle
from evaloop.models.base import ModelUnavailableError
from evaloop.tasks import TaskFactory

from conftest import MBPP_STYLE_PROMPT, SUMMARIZATION_CONFIG, GENERATION_CONFIG

OPENROUTER = "https://openrouter.ai/api/v1"


def make_prompts(n):
    return [{"task_id": f"Mbpp/{i}", "prompt": f"p{i}"} for i in range(n)]


def make_result(task_ids, **overrides):
    result = {
        "model": "OpenAICompatibleLLM(model_name=m)",
        "max_cycles": 10,
        "summary_language": "english",
        "max_tasks": None,
        "subset_seed": 0,
        "prompt_results": [{"task_id": t, "cycles": [], "successful_cycles": 0} for t in task_ids],
    }
    result.update(overrides)
    return result


class TestSelectTaskSubset:
    def test_none_keeps_everything(self):
        prompts = make_prompts(5)
        assert select_task_subset(prompts, None, 0) == prompts

    def test_larger_than_dataset_keeps_everything(self):
        prompts = make_prompts(5)
        assert select_task_subset(prompts, 10, 0) == prompts

    def test_same_seed_same_subset_in_dataset_order(self):
        prompts = make_prompts(50)
        first = select_task_subset(prompts, 10, 7)
        assert first == select_task_subset(prompts, 10, 7)
        assert len(first) == 10
        positions = [prompts.index(p) for p in first]
        assert positions == sorted(positions)

    def test_different_seed_different_subset(self):
        prompts = make_prompts(50)
        assert select_task_subset(prompts, 10, 1) != select_task_subset(prompts, 10, 2)


class TestCheckpoint:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_checkpoint(tmp_path / "none.json", make_result(["a"])) is None

    def test_round_trip(self, tmp_path):
        path = tmp_path / "ckpt.json"
        result = make_result(["a", "b"])
        result["prompt_results"][0]["successful_cycles"] = 2
        save_checkpoint(path, result, 2, [{"task_id": "a", "prompt": "next"}])

        state = load_checkpoint(path, make_result(["a", "b"]))
        assert state["completed_cycle"] == 2
        assert state["next_prompts"] == [{"task_id": "a", "prompt": "next"}]
        assert state["model_result"]["prompt_results"][0]["successful_cycles"] == 2
        assert not (tmp_path / "ckpt.json.tmp").exists()

    def test_non_json_values_are_stringified(self, tmp_path):
        path = tmp_path / "ckpt.json"
        result = make_result(["a"])
        result["prompt_results"][0]["cycles"].append({"details": {1, 2}})
        save_checkpoint(path, result, 1, [])
        json.loads(path.read_text())

    @pytest.mark.parametrize("key,value", [("summary_language", "spanish"), ("max_cycles", 5), ("subset_seed", 3)])
    def test_rejects_other_configuration(self, tmp_path, key, value):
        path = tmp_path / "ckpt.json"
        save_checkpoint(path, make_result(["a"]), 1, [])
        with pytest.raises(ValueError, match=key):
            load_checkpoint(path, make_result(["a"], **{key: value}))

    def test_rejects_other_task_set(self, tmp_path):
        path = tmp_path / "ckpt.json"
        save_checkpoint(path, make_result(["a", "b"]), 1, [])
        with pytest.raises(ValueError, match="task set"):
            load_checkpoint(path, make_result(["a", "c"]))


class TestEndpointConfig:
    def make_config(self, tmp_path, **kwargs):
        return EvaluationConfig(model="google/gemma-3-12b-it", output_dir=str(tmp_path), **kwargs)

    def test_endpoint_routes_any_model_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        config = self.make_config(tmp_path, api_base_url=OPENROUTER, api_key_env="OPENROUTER_API_KEY")
        model_config = config.get_model_config()
        assert model_config["type"] == "openai_compatible"
        assert model_config["model_id"] == "google/gemma-3-12b-it"
        assert model_config["base_url"] == OPENROUTER
        assert model_config["api_key"] == "sk-test"
        assert model_config["max_tokens"] == config.max_new_tokens == 1024
        assert model_config["temperature"] == 0.0
        assert "extra_body" not in model_config

    def test_provider_pinning(self, tmp_path):
        config = self.make_config(tmp_path, api_base_url=OPENROUTER, api_provider="deepinfra/bf16, other")
        assert config.get_model_config()["extra_body"] == {
            "provider": {"order": ["deepinfra/bf16", "other"], "allow_fallbacks": False}
        }

    def test_without_endpoint_routing_is_unchanged(self, tmp_path):
        config = EvaluationConfig(model="gpt-4o-mini", output_dir=str(tmp_path))
        assert config.get_model_config()["type"] == "openai"

    def test_resume_needs_experiment_name(self, tmp_path):
        with pytest.raises(ValueError, match="experiment_name"):
            self.make_config(tmp_path, resume=True)
        self.make_config(tmp_path, resume=True, experiment_name="run1")

    @pytest.mark.parametrize("kwargs", [{"api_concurrency": 0}, {"max_tasks": 0}])
    def test_validation(self, tmp_path, kwargs):
        with pytest.raises(ValueError):
            self.make_config(tmp_path, **kwargs)


class TestOpenAICompatibleLLM:
    @pytest.fixture
    def llm_class(self, monkeypatch):
        pytest.importorskip("openai")
        from evaloop.models import closed_source
        from evaloop.models.openai_compatible import OpenAICompatibleLLM

        calls = []

        def fake_generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return self.fake_responses.get(prompt, f"answer to {prompt}")

        monkeypatch.setattr(closed_source.OpenAILLM, "generate", fake_generate)
        OpenAICompatibleLLM.calls = calls
        return OpenAICompatibleLLM

    def make_llm(self, llm_class, responses=None, **config):
        llm = llm_class({"name": "m", "model_id": "m", "api_key": "sk-test", "base_url": OPENROUTER,
                         "endpoint_retry_delay": 0, **config})
        llm.fake_responses = responses or {}
        return llm

    def test_requires_api_key(self, llm_class):
        with pytest.raises(ValueError, match="API key"):
            llm_class({"name": "m", "model_id": "m", "api_key": None, "base_url": OPENROUTER})

    def test_batch_keeps_input_order(self, llm_class):
        llm = self.make_llm(llm_class, max_concurrency=4)
        prompts = [f"q{i}" for i in range(20)]
        assert llm.batch_generate(prompts) == [f"answer to q{i}" for i in range(20)]

    def test_extra_body_sent_with_every_request(self, llm_class):
        body = {"provider": {"order": ["deepinfra/bf16"], "allow_fallbacks": False}}
        llm = self.make_llm(llm_class, extra_body=body)
        llm.batch_generate(["a", "b"])
        assert all(kwargs["extra_body"] == body for _, kwargs in llm_class.calls)

    def test_endpoint_error_raises_after_retries(self, llm_class):
        llm = self.make_llm(llm_class, {"bad": "Error generating response: 402 insufficient credits"},
                            endpoint_retries=2)
        with pytest.raises(ModelUnavailableError, match="402"):
            llm.generate("bad")
        assert len(llm_class.calls) == 3

    def test_empty_response_message_is_model_output(self, llm_class):
        message = "Error: Still received empty response after multiple attempts."
        llm = self.make_llm(llm_class, {"q": message})
        assert llm.generate("q") == message


class _BatchModel:
    def __init__(self, error):
        self.error = error

    def generate(self, prompt):
        raise self.error

    def batch_generate(self, prompts):
        raise self.error


class _SequentialModel:
    def __init__(self, error):
        self.error = error

    def generate(self, prompt):
        raise self.error


class TestCycleEndpointFailures:
    def make_tasks(self):
        return [TaskFactory().create_task(dict(c)) for c in (GENERATION_CONFIG, SUMMARIZATION_CONFIG)]

    def prompts(self, n):
        return [{"task_id": f"Mbpp/{i}", "prompt": MBPP_STYLE_PROMPT} for i in range(n)]

    @pytest.mark.parametrize("model_class,n", [(_BatchModel, 2), (_SequentialModel, 1)])
    def test_endpoint_failure_stops_the_run(self, model_class, n):
        runner = BatchEvaluationCycle(model_class(ModelUnavailableError("down")), self.make_tasks())
        with pytest.raises(ModelUnavailableError):
            runner.run_single_cycle(self.prompts(n), 1)

    @pytest.mark.parametrize("model_class,n", [(_BatchModel, 2), (_SequentialModel, 1)])
    def test_other_errors_still_fail_the_cycle(self, model_class, n):
        runner = BatchEvaluationCycle(model_class(RuntimeError("bad output")), self.make_tasks())
        results = runner.run_single_cycle(self.prompts(n), 1)["cycle_results"]
        assert all("task_1_error" in r for r in results)
