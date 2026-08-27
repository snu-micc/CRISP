import json
import unittest
from pathlib import Path

from crisp.audit import audit_generated_code
from crisp.prompts import TASKS, code_prompt
from crisp.workflow import parse_numbered_rules


class OfflineWorkflowTests(unittest.TestCase):
    def test_numbered_rule_parser_accepts_multiline_and_inline_lists(self):
        multiline = "\n".join(f"{index}: rule {index}" for index in range(1, 11))
        inline = ", ".join(f"{index}: rule {index}" for index in range(1, 11))
        expected = [f"rule {index}" for index in range(1, 11)]
        self.assertEqual(parse_numbered_rules(multiline), expected)
        self.assertEqual(parse_numbered_rules(inline), expected)

    def test_static_gate_accepts_one_structure_function(self):
        code = "def rule_01_example(structure):\n    return float(len(structure))\n"
        result = audit_generated_code(code, "rule_01_example")
        self.assertTrue(result.passed)

    def test_static_gate_rejects_external_access(self):
        code = "def rule_01_example(structure):\n    return open('/tmp/value').read()\n"
        result = audit_generated_code(code, "rule_01_example")
        self.assertFalse(result.passed)
        self.assertIn("forbidden_call:open", result.issues)

        network_code = "def rule_01_example(structure):\n    return MPRester().search()\n"
        network_result = audit_generated_code(network_code, "rule_01_example")
        self.assertFalse(network_result.passed)
        self.assertIn("forbidden_access:MPRester", network_result.issues)

    def test_all_public_tasks_render_code_prompts(self):
        self.assertEqual(
            set(TASKS),
            {
                "synthesizability",
                "formation_energy",
                "ionic_conductivity",
                "shear_modulus",
            },
        )
        for spec in TASKS.values():
            rendered = code_prompt(spec, 1, "example", "Example chemical rule")
            self.assertIn("rule_01_example", rendered)
            self.assertIn("Example chemical rule", rendered)

    def test_published_catalog_has_fifty_ordered_rule_ids(self):
        root = Path(__file__).resolve().parents[1]
        catalog = json.loads(
            (root / "catalogs" / "synthesizability_rules.json").read_text(encoding="utf-8")
        )
        self.assertEqual(catalog["dimension"], 50)
        self.assertEqual(
            [row["rule_id"] for row in catalog["rules"]],
            [f"R{index}" for index in range(1, 51)],
        )
        self.assertTrue(all(row["name"] for row in catalog["rules"]))
        self.assertTrue(all(row["operational_summary"] for row in catalog["rules"]))


if __name__ == "__main__":
    unittest.main()
