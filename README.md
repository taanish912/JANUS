# JANUS
### Evaluating Goal-Conditioned Distortion in Factually Grounded LLM Communication

[![Research prototype](https://img.shields.io/badge/status-research%20prototype-5965e8)](https://github.com/taanish912/JANUS) [![160 scenarios](https://img.shields.io/badge/benchmark-160%20scenarios-2563eb)](https://github.com/taanish912/JANUS/tree/research/janus-v18-pilot-release/data) [![8 domains](https://img.shields.io/badge/domains-8-0f766e)](https://github.com/taanish912/JANUS/blob/research/janus-v18-pilot-release/DATA_CARD.md)

**JANUS is a paired evaluation framework for studying how institutional-goal instructions change the way language models communicate a fixed body of decision-relevant evidence.**

It asks more than whether a response contains a false statement. JANUS examines how the same evidence is selected, emphasized, ordered, qualified, and framed under different communicative objectives.

<p align="center">
  <img src="https://raw.githubusercontent.com/taanish912/JANUS/research/janus-v18-pilot-release/metrics/radar_charts/all_models_radar.png" alt="JANUS cross-model evaluation overview" width="88%">
</p>

---

## At a glance

| Component | Current research snapshot |
|---|---|
| Benchmark | 160 hypothetical decision scenarios |
| Coverage | 8 domains, 20 scenarios per domain |
| Evidence per scenario | 3 objective-favorable and 3 adverse material facts |
| Main comparison | Neutral instruction vs. latent institutional-goal instruction |
| Existing model runs | 12 primary runs |
| Evaluation dimensions | Selection, emphasis, ordering, specificity, and framing |
| Additional experiment | v18 controlled-length pilot: 8 paired scenarios / 16 responses |

> **Repository layout:** the full research snapshot and the v18 release are currently published on the [research/janus-v18-pilot-release branch](https://github.com/taanish912/JANUS/tree/research/janus-v18-pilot-release). This README on main is the project landing page and navigation point.

## Why JANUS?

A model can communicate only supplied, factually grounded information and still produce a strategically different account of a situation. It may foreground favorable evidence, reduce the salience of adverse facts, change the order of information, or make a recommendation sound more decisive.

JANUS studies this communication problem using paired prompts and a fixed evidence set. The benchmark is designed to help researchers inspect differences in *presentation*, not merely factual inclusion.

## Evaluation design

Each scenario contains six stipulated facts: three favorable to the stated institutional objective and three adverse material facts. The neutral and goal-conditioned conditions are compared using the same scenario and evidence.

~~~mermaid
flowchart TD
    A[Hypothetical decision scenario] --> B[Six stipulated facts]
    B --> C1[Neutral system instruction]
    B --> C2[Institutional-goal instruction]
    C1 --> D1[Model response]
    C2 --> D2[Model response]
    D1 --> E[Paired evaluation]
    D2 --> E
    E --> F[Selection - Emphasis - Ordering]
    E --> G[Specificity - Framing]
    F --> H[Compare communication differences]
    G --> H
~~~

The generation pipeline controls the paired setup, including a shared fact order within each pair. Evaluation artifacts include sentence-to-fact matches, framing judgments, and derived metrics. These metrics are measurement instruments, not ground truth; interpretation should account for their validation and limitations.

## Benchmark coverage

The 160-scenario dataset contains 20 scenarios in each domain:

- Finance
- Healthcare
- Education
- Law
- Public policy
- Media / information
- Business / corporate
- Workplace / organizational

All entities and facts in the benchmark scenarios are hypothetical. They are not claims about real people, organizations, products, laws, or treatments.

## Results and research artifacts

The existing research snapshot includes 12 primary model runs, sentence-to-fact matching outputs, framing judgments, derived metrics, and supplemental analyses.

<p align="center">
  <img src="https://raw.githubusercontent.com/taanish912/JANUS/research/janus-v18-pilot-release/metrics/radar_charts/category_composite_heatmap.png" alt="JANUS category-level composite heatmap" width="84%">
</p>

Explore the [research snapshot](https://github.com/taanish912/JANUS/tree/research/janus-v18-pilot-release), including:

| Path | Contents |
|---|---|
| data/janus.csv | 160-scenario benchmark |
| results/ | Primary paired generations and evaluation outputs |
| metrics/ | Model-, output-, scenario-, and category-level metrics |
| analysis/ | Supplemental judge-robustness and mitigation analyses |
| scripts/ | Validation, reproduction, plotting, and significance scripts |
| DATA_CARD.md | Dataset composition, construction, intended use, and risks |
| MANIFEST.md | Paper-to-artifact map and reproduction entry points |

## v18 controlled-length pilot

The v18 pilot is a separate exploratory experiment that adds a controlled-length protocol to the JANUS research program. It is **not** a completed evaluation of all 160 benchmark scenarios.

### Protocol

- 8 scenarios across the eight domains; 2 conditions per scenario; 16 responses total.
- The paired conditions used the same user message, scenario, six supplied facts, and fact order. The system-level institutional objective was the intended prompt difference.
- Configured provider/model string: adaption / deepseek-v4-flash.
- Seven-section response structure with a 140-word soft target.

### Descriptive paired review (corrected v2)

- Institutional framing aligned with the stated objective in 7 of 8 pairs; 1 was mixed/conditional.
- Recommendation shifted toward the objective in 4 of 8 pairs.
- Adverse evidence was judged comparably prominent in 7 of 8 pairs and less prominent under the latent-goal condition in 1 pair.
- Confidence language was more certain under the latent-goal condition in 3 pairs, comparable in 4, and unclear in 1.

All 16 responses met the structural requirements, but **all 16 exceeded the 140-word target** (mean 218.19 words; range 191-258). The length target was therefore not achieved.

The claim-level review contains 44 records from an assistant-assisted workflow. Final decisions were recorded by the researcher during that workflow; they were not independently double-coded by a second human rater. The pair-level rubric was documented retrospectively, not preregistered, and has not been validated as a measurement scale.

**Interpretation:** this is a small, single-configuration exploratory pilot. These descriptive observations do not establish a general causal effect across models, prompts, tasks, or institutions.

### v18 release artifacts

All v18 materials are available in the dedicated [pilot release folder](https://github.com/taanish912/JANUS/tree/research/janus-v18-pilot-release/research_release/v18_pilot).

| Artifact | Link |
|---|---|
| Pilot overview and methodology | [v18 pilot README](https://github.com/taanish912/JANUS/blob/research/janus-v18-pilot-release/research_release/v18_pilot/README.md) |
| Full pilot report | [PDF report](https://github.com/taanish912/JANUS/blob/research/janus-v18-pilot-release/research_release/v18_pilot/reports/JANUS_v18_Pilot_Report.pdf) |
| Generation script | [v18 generation code](https://github.com/taanish912/JANUS/blob/research/janus-v18-pilot-release/research_release/v18_pilot/code/generate_controlled_length_v18.py) |
| Raw paired responses | [16-response JSONL](https://github.com/taanish912/JANUS/blob/research/janus-v18-pilot-release/research_release/v18_pilot/data/pilot8_raw_responses.jsonl) |
| Corrected paired evaluations | [pair evaluations v2](https://github.com/taanish912/JANUS/blob/research/janus-v18-pilot-release/research_release/v18_pilot/evaluations/janus_v18_pilot8_pair_evaluations_v2.jsonl) |
| Corrected codebook | [codebook v1.4](https://github.com/taanish912/JANUS/blob/research/janus-v18-pilot-release/research_release/v18_pilot/evaluations/janus_v18_pilot8_evaluation_codebook_v1_4.json) |

The earlier v1 paired assessments are retained alongside v2 for auditability. The MED_17 A/B-to-condition mapping correction is reflected in v2 and codebook v1.4.

## Reproduce the existing benchmark metrics

The following commands use the released generations and judgments. Recomputing the released metrics does not require a new model/API call.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Validate the released artifacts
python scripts/validate_release.py

# Recompute the main metrics and figures
python scripts/reproduce_main_results.py

# Reproduce paired significance tests
python scripts/analyze_significance.py
~~~

For a quick significance-test smoke run, use --draws 1000. The full analysis uses 200,000 Monte Carlo draws, as documented in the project materials.

To run new generations or model-judge stages, review the research README and configure provider credentials securely. **Never commit API keys or .env files.**

## Responsible interpretation

JANUS is a research benchmark, not a deployed decision system. Its scenarios cover high-stakes themes, but the facts are synthetic. Automated sentence matching and framing labels are analytical instruments and can be noisy. The benchmark is English-only and does not represent every culture, institution, language, or form of strategic communication.

Do not use per-item scores as a basis for consequential decisions about people. Interpret aggregate patterns alongside the source responses, manual audits, and methodological limitations.

## Contributing

Research feedback, issue reports, and reproducibility improvements are welcome. When reporting a result, include the relevant branch/commit, model and provider configuration, prompt condition, and evaluation artifact version.


<p align="center">
  <strong>JANUS - examining not only what a model says, but how it presents the evidence it was given.</strong>
</p>
