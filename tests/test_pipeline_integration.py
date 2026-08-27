import re
import tempfile
import unittest
from pathlib import Path

from crisp.workflow import _consolidate, _discover, _embed_and_cluster, _generate_code


class _ModelObject:
    def __init__(self, **values):
        self.__dict__.update(values)

    def model_dump(self, mode="json"):
        return dict(self.__dict__)


class _Responses:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        prompt = kwargs["input"]
        if "suggest 10 key" in prompt:
            text = "\n".join(f"{index}: chemical rule {index}" for index in range(1, 11))
        elif "RULE TEXTS:" in prompt:
            text = "Quantify the local coordination environment and composition."
        else:
            match = re.search(r"Name the function `([^`]+)`", prompt)
            if not match:
                raise AssertionError("Expected function name was not rendered")
            text = f"def {match.group(1)}(structure):\n    return float(len(structure))"
        return _ModelObject(
            id=f"response-{self.calls}",
            model=kwargs["model"],
            output_text=text,
            usage={"input_tokens": 1, "output_tokens": 1},
        )


class _Embeddings:
    def create(self, **kwargs):
        data = []
        for index, text in enumerate(kwargs["input"]):
            seed = sum(ord(character) for character in text)
            vector = [((seed + 17 * offset) % 101) / 100 for offset in range(8)]
            data.append(_ModelObject(index=index, embedding=vector))
        return _ModelObject(model=kwargs["model"], data=data, usage={"total_tokens": 1})


class _Client:
    def __init__(self):
        self.responses = _Responses()
        self.embeddings = _Embeddings()


class OfflinePipelineIntegrationTests(unittest.TestCase):
    def test_all_compilation_stages_with_fake_api(self):
        config = {
            "task": "synthesizability",
            "discovery_model": "discovery-test",
            "embedding_model": "embedding-test",
            "consolidation_model": "consolidation-test",
            "code_generation_model": "code-test",
            "discovery_calls": 2,
            "rules_per_call": 10,
            "clusters": 3,
            "pca_dimensions": 4,
            "kmeans_n_init": 2,
            "kmeans_max_iter": 50,
            "random_seed": 42,
            "workers": 2,
        }
        client = _Client()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rules, records = _discover(client, config, output)
            assignments, summaries = _embed_and_cluster(client, rules, config, output)
            consolidated = _consolidate(client, assignments, summaries, config, output)
            manifest = _generate_code(client, consolidated, config, output)

            self.assertEqual(len(records), 2)
            self.assertEqual(len(rules), 20)
            self.assertEqual(len(assignments), 20)
            self.assertEqual(len(summaries), 3)
            self.assertEqual(len(consolidated), 3)
            self.assertEqual(len(manifest), 3)
            self.assertTrue(all(row["static_audit"]["passed"] for row in manifest))
            self.assertTrue((output / "stages" / "05_generated_code_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
