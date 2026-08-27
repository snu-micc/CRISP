"""Frozen, dataset-blind prompt templates for CRISP rule compilation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskPrompts:
    slug: str
    label: str
    discovery: str
    consolidation_system: str
    consolidation_user: str
    code_system: str
    code_user: str


COMMON_CODE_REQUIREMENTS = """Requirements:
1. The function must take exactly one required argument named `structure`.
2. It must return one finite Python float. Use 0.0 only as a documented failure fallback.
3. Use explicit, auditable calculations from the crystal structure, its composition, and fixed tabulated elemental properties available through standard scientific Python libraries.
4. Do not access files, environment variables, material identifiers, target measurements, labels, dataset statistics, fitted parameters, predictions, or residuals.
5. Do not call a graph neural network, pretrained property model, surrogate predictor, interatomic potential, or another machine-learning model.
6. Do not encode chemical-family names or benchmark-specific lookup tables.
7. Prefer concise logic with bounded neighbor searches; avoid dense grids or supercells.
8. Name the function `rule_{rule_id:02d}_{slug}`.
9. Imports will be supplied by a later, human-reviewed execution harness.
10. Output only the complete function definition without Markdown fences, imports, tests, or explanation."""


def _task(
    *,
    slug: str,
    label: str,
    expertise: str,
    target: str,
    exclusions: str,
) -> TaskPrompts:
    discovery = f"""You are an expert in {expertise}. Given any inorganic material with crystal information (CIF files), suggest 10 key structure- or composition-derived features that are important indicators of {target}.

The indicators must be expressible from the crystal structure, its composition, and fixed tabulated elemental properties. {exclusions}

Please provide the 10 rules in the following format:

1: (reason), 2: (reason), ..., and keep each explanation under 100 words.

DO NOT provide any other information or explanation."""
    consolidation_system = (
        f"You are an expert in {expertise}. Consolidate semantically related "
        "chemical rules without using material examples, target values, dataset "
        "statistics, labels, or prediction results."
    )
    consolidation_user = f"""The rule texts below belong to one semantic cluster generated for {target}.

Synthesize one representative chemistry-rule paragraph that captures their shared rationale. Remove repetition and material-specific details while preserving common mechanistic content. State what structure-, composition-, or tabulated-element-derived property should be quantified from a CIF-derived crystal structure. {exclusions} Do not mention clustering, language models, datasets, labels, or performance. Do not invent numeric thresholds. Return only the representative paragraph.

RULE TEXTS:
{{rule_texts}}"""
    code_system = f"You are an expert in {expertise} and computational materials science."
    code_user = f"""Based only on the rule content below, write one Python function that computes the corresponding scalar descriptor from a pymatgen Structure parsed from a CIF file.

{{common_requirements}}

Target-specific restriction: {exclusions}

RULE CONTENT:
{{rule_text}}"""
    return TaskPrompts(
        slug=slug,
        label=label,
        discovery=discovery,
        consolidation_system=consolidation_system,
        consolidation_user=consolidation_user,
        code_system=code_system,
        code_user=code_user,
    )


SYNTHESIZABILITY = TaskPrompts(
    slug="synthesizability",
    label="inorganic-crystal synthesizability",
    discovery="""You are an expert in inorganic chemistry. Given any inorganic material with crystal information (CIF files), suggest 10 key features that are important indicators of its synthesizability.

Please provide the 10 rules in the following format:
1: (reason), 2: (reason), ..., and must keep each explanation under 100 words.

DO NOT provide any other information or explanation.""",
    consolidation_system=(
        "You are an expert in inorganic chemistry and crystal chemistry. Consolidate "
        "semantically related chemical rules without using structures, material "
        "identifiers, labels, dataset statistics, or prediction results."
    ),
    consolidation_user="""The rule texts below belong to one semantic cluster generated for inorganic-crystal synthesizability.

Synthesize one representative chemistry-rule paragraph that captures their shared chemical rationale. Remove repetition and overly material-specific details while preserving distinct mechanistic content common to the cluster. State what structural or physicochemical property should be quantified from a CIF-derived crystal structure. Do not mention clustering, language models, datasets, labels, or model performance. Do not invent numeric thresholds. Return only the representative paragraph.

RULE TEXTS:
{rule_texts}""",
    code_system=(
        "You are an expert in inorganic chemistry, crystal chemistry, materials "
        "informatics, and computational materials science."
    ),
    code_user="""Based only on the rule content below, write one Python function that computes the corresponding scalar descriptor from a pymatgen Structure parsed from a CIF file.

{common_requirements}

Target-specific restriction: Do not use synthesis labels, database membership as a label, or outputs from pretrained synthesizability models or other learned surrogates.

RULE CONTENT:
{rule_text}""",
)

FORMATION_ENERGY = _task(
    slug="formation_energy",
    label="DFT formation energy per atom",
    expertise="inorganic chemistry, solid-state thermodynamics, crystal chemistry, and materials informatics",
    target="DFT formation energy per atom relative to elemental reference states",
    exclusions=(
        "Do not use formation energy, total energy, energy above hull, decomposition "
        "energy, material-specific DFT results, database-computed target properties, "
        "or predictions from pretrained or surrogate models."
    ),
)

IONIC_CONDUCTIVITY = _task(
    slug="ionic_conductivity",
    label="room-temperature ionic conductivity",
    expertise="solid-state ionics, solid-state electrolyte batteries, inorganic crystal chemistry, and materials informatics",
    target="room-temperature ionic conductivity of inorganic solid-state electrolytes",
    exclusions=(
        "Do not use conductivity measurements, fitted transport parameters, "
        "database-computed target properties, or predictions from pretrained or "
        "surrogate models."
    ),
)

SHEAR_MODULUS = _task(
    slug="shear_modulus",
    label="isotropic Voigt-Reuss-Hill shear modulus",
    expertise="inorganic chemistry, solid-state mechanics, crystal chemistry, and materials informatics",
    target="the isotropic Voigt-Reuss-Hill shear modulus of inorganic crystals",
    exclusions=(
        "Do not use elastic tensors, elastic constants, bulk or shear moduli, "
        "stress-strain calculations, material-specific mechanics calculations, "
        "or predictions from pretrained or surrogate models."
    ),
)

TASKS = {
    item.slug: item
    for item in (
        SYNTHESIZABILITY,
        FORMATION_ENERGY,
        IONIC_CONDUCTIVITY,
        SHEAR_MODULUS,
    )
}


def get_task_prompts(slug: str) -> TaskPrompts:
    """Return a frozen prompt set by task slug."""
    try:
        spec = TASKS[slug]
    except KeyError as exc:
        raise ValueError(f"Unknown task {slug!r}; choose from {sorted(TASKS)}") from exc
    return spec


def code_prompt(spec: TaskPrompts, rule_id: int, slug: str, rule_text: str) -> str:
    """Fill the shared code-generation template for one consolidated rule."""
    return spec.code_user.format(
        common_requirements=COMMON_CODE_REQUIREMENTS.format(rule_id=rule_id, slug=slug),
        rule_text=rule_text,
    )
