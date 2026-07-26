# Causal Inference Engine

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikitlearn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-4051B5)
![Streamlit](https://img.shields.io/badge/Streamlit-interface-red?logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Estimate what actually caused what, with the assumptions, robustness checks, and plain-English caveats attached.

## The Problem

You send a discount to some customers. The ones who got it spend more. Did the discount work?

Probably not — or at least, not by nearly as much as it looks. You almost certainly sent it to customers who were already your biggest spenders, so the comparison credits the discount with behaviour those customers would have shown anyway. The correlation is completely real and completely misleading.

An A/B test solves this by assigning at random, but you often cannot run one. The policy already rolled out. The data already exists. Randomising would be unethical, illegal, or impossible.

## What This Does

This estimates causal effects from data that was never randomised, and — more importantly — tells you when it cannot.

- It **draws the assumptions as a diagram** and sorts each variable by the role it plays, because controlling for the wrong one silently destroys the answer.
- It **checks whether the comparison is even possible** before producing any number.
- It **estimates the effect five different ways** and shows them side by side, since agreement across methods that rely on different assumptions is far stronger evidence than any single number.
- It **tries to break its own result** — including replacing the treatment with random noise, which any pipeline that manufactures signal will fail.
- It **reports how fragile the answer is**: how strong an unmeasured factor would have to be to erase it.
- It **says all of this in plain English**, and exports a PDF that leads with whether the analysis can be trusted.

Think of it as a colleague who, before telling you the number, tells you whether the number means anything.

---

## Does it actually work?

Every claim above is checked against data where the true effect is known by construction, because a causal estimate on real data cannot be graded — the right answer is unknown by definition.

```
python -m validation.ground_truth
```

```
dataset        method                           true     naive  estimate  bias cut  expected  ok
----------------------------------------------------------------------------------------------------
confounded     psm                             2.000     5.992     2.029     99.3%  recover   yes
confounded     ipw                             2.000     5.992     1.715     92.9%  recover   yes
heterogeneous  x_learner                       1.895     1.961     1.900     91.7%  recover   yes
did_panel      did                             3.000     7.214     2.989     99.7%  recover   yes
instrumental   ipw (cannot fix this)           1.500     4.122     4.125     -0.1%  fail      yes
instrumental   iv                              1.500     4.122     1.665     93.7%  recover   yes
no_effect      psm                             0.000     4.198     0.031     99.3%  recover   yes
no_effect      ipw                             0.000     4.198     0.102     97.6%  recover   yes
realistic      psm                             0.150     1.263     1.068     17.5%  improve   yes
realistic      ipw                             0.150     1.263     0.825     39.4%  improve   yes
----------------------------------------------------------------------------------------------------
10 of 10 behaved as expected
```

Three rows carry most of the weight:

**The no-effect case.** The treatment does nothing at all, and a raw comparison confidently reports **4.20**. Both estimators return approximately zero and mark it non-significant. A tool that always finds an effect is worse than no tool, so this case is part of the suite rather than an afterthought.

**The instrumental case.** The confounder is deliberately absent from the data. Adjustment stays wrong at **4.13** no matter how carefully it is applied, and only the instrument recovers **1.67**. This is graded as a *successful* demonstration that adjustment has limits.

**The realistic case.** A small effect, a confounder observed only as a noisy proxy, and another missing entirely. The estimators cut the bias but land at 0.83–1.07 against a true 0.15, with intervals that confidently exclude the truth. That is the normal outcome on observational data, not a defect.

Effect variation is recovered too, including a segment the treatment **harms**:

```
  A: estimated +5.02 against a true +5.00
  B: estimated +1.49 against a true +1.50
  C: estimated -1.99 against a true -2.00  (harmed)
  correlation with the planted per-unit effect: 0.997
```

The average effect is a positive +1.9, which on its own would hide that a quarter of the population is made worse off.

---

## Screenshots

### Results — every method against the known true effect

![Results](docs/ui-results.png)

### Assumptions — checked, and explained, before any number appears

![Assumptions](docs/ui-assumptions.png)

### Effect sizes stay behind the assumptions page

![Gating](docs/ui-gating.png)

### The causal graph decides what may be adjusted for

![Causal graph](docs/ui-graph.png)

---

## Quick Start

Requires Python 3.11+.

```bash
git clone https://github.com/Emart29/causal-inference-engine
cd causal-inference-engine
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
pip install -e .

# Confirm the estimators recover known effects
python -m validation.ground_truth

# Work through the examples
python -m examples.marketing_uplift
python -m examples.policy_effect
python -m examples.ab_extension

# Launch the interface
streamlit run ui/app.py
```

With Docker, against the shared infrastructure stack:

```bash
docker compose up                      # interface on 8501
docker compose --profile validate up   # run the ground truth suite and exit
```

---

## The methods, and what each one actually estimates

This distinction matters more than it looks. Reporting an effect measured on one group as though it applied to another is a misstatement, not a rounding error, so every result carries its estimand.

| Method | Use when | Estimates |
| --- | --- | --- |
| **Propensity score matching** | you measured the confounders and want a like-for-like comparison | the effect **on those who were treated** |
| **Inverse probability weighting** | you measured the confounders and want a population answer | the effect **across everyone** |
| **S / T / X-learners** | you need to know who responds, not just whether it worked | the effect **per individual** |
| **Difference-in-differences** | a policy switched on at a known time for one group | the effect **on the treated group after the change** |
| **Instrumental variables** | the confounder is not in your data at all | the effect **on those the instrument moved** — nobody else |

### The mistake this exists to prevent

Not every variable should be controlled for:

- a **confounder** causes both treatment and outcome — adjusting removes bias,
- a **mediator** sits on the causal path — adjusting removes part of the effect you are trying to measure,
- a **collider** is caused by both — adjusting creates an association that does not exist.

The engine classifies each covariate and drops mediators and colliders from the adjustment set automatically, in plain English, before anything is estimated. This is the single most common way a competent analysis produces a confidently wrong answer.

---

## How it can still mislead you

No statistical method rescues a bad research design. These are the failure modes this tool cannot fix:

**Unmeasured confounding.** Everything except the instrumental variables path assumes you measured the variables that drive both treatment and outcome. If one is missing, the estimate is biased and no amount of adjustment corrects it — as the instrumental row in the validation table shows.

**A wrong graph.** The graph is an assumption. Causal structure is not identifiable from observational data without further assumptions, so the default here is a documented convention, not a discovery. Get it wrong and the adjustment set is wrong.

**Weak instruments.** Below a first-stage F of about 10, two-stage least squares is badly biased. The engine reports the statistic and flags it.

**The exclusion restriction.** That an instrument affects the outcome *only* through the treatment cannot be tested from data — a violation looks exactly like a real effect. It is recorded as an explicit untestable claim.

**Violated parallel trends.** Difference-in-differences relies on the groups having moved together beforehand. That is testable, and it is tested.

**Small effects in noisy data.** See the realistic row above: the methods improve on the naive comparison and still miss the answer.

The real-world benchmark makes the point better than any caveat. On the [LaLonde job training data](https://users.nber.org/~rdehejia/), where a randomised trial puts the true effect at **+$1,794**, pairing the participants with a population survey makes a raw comparison report **−$15,578** — wrong by more than seventeen thousand dollars, and in the wrong direction. Matching recovers most of it; weighting barely helps. The assumption checks refuse the analysis outright, because 47% of units fall outside common support:

> This data cannot support a credible causal estimate.

That refusal is the feature.

---

## Interface

Five steps, ordered so the conditions arrive before the number:

| Step | What happens |
| --- | --- |
| **Data and question** | choose a dataset with a known effect, or upload your own; name treatment, outcome, covariates |
| **Causal graph** | review and edit the assumed structure; see each variable's role and the resulting adjustment set |
| **Assumptions** | overlap, positivity, balance, and sample size, each with a plain-English reading |
| **Results** | every method side by side, robustness checks, sensitivity, and who the effect falls on |
| **Report** | the assembled document, exportable as a PDF |

Results are deliberately withheld until the assumptions page has been opened. An effect size seen first tends to be remembered regardless of what follows.

---

## Project Layout

```text
causal-inference-engine/
├── datasets/
│   ├── generate.py       # six generators, each with its true effect attached
│   └── lalonde.py        # the real benchmark, graded against a randomised trial
├── discovery/
│   ├── dag_builder.py    # the causal graph and its validation
│   ├── confounder.py     # covariate roles and the safe adjustment set
│   └── assumption_check.py
├── estimation/
│   ├── base.py           # the estimand-carrying result type
│   ├── propensity.py     # matching and weighting
│   ├── uplift.py         # per-unit effects and targeting
│   ├── did.py            # difference-in-differences, with a trends test
│   ├── iv.py             # two-stage least squares
│   └── run.py            # orchestration
├── validation/
│   ├── refutation.py     # attempts to discredit an estimate
│   ├── sensitivity.py    # how much hidden confounding would overturn it
│   └── ground_truth.py   # the suite behind the table above
├── reporting/
│   ├── interpret.py      # the sentences
│   ├── visualizer.py     # the charts
│   ├── report.py         # assembly, in reading order
│   └── pdf_export.py
├── ui/app.py
└── examples/
```

---

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `POSTGRES_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/causal` | stored analyses |
| `MINIO_ENDPOINT` | `localhost:9000` | uploaded data and exported reports |
| `MINIO_BUCKET` | `causal-artifacts` | bucket for both |
| `DEFAULT_N_BOOTSTRAP` | `200` | resamples for methods without a closed-form interval |
| `RANDOM_SEED` | `42` | so a rerun reproduces the same numbers |

---

## Part of the ML Platform

The fifth project in a connected series, and the first that answers a business question rather than serving a model:

1. **[ml-feature-store](https://github.com/Emart29/ml-feature-store)** — define a feature once, serve it consistently for training and inference.
2. **[pipeline-lineage-tracker](https://github.com/Emart29/pipeline-lineage-tracker)** — trace any prediction back to the data and code that produced it.
3. **[ml-canary-deploy](https://github.com/Emart29/ml-canary-deploy)** — ship model updates to a slice of traffic with automatic rollback.
4. **[realtime-anomaly-detection](https://github.com/Emart29/realtime-anomaly-detection)** — catch problems in streaming data within seconds.
5. **causal-inference-engine** — decide whether the thing you did actually worked.
