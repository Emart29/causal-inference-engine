"""The LaLonde job training benchmark, the standard real-data test for these methods.

The National Supported Work programme randomised people into a job training
scheme and measured their later earnings, so the experimental comparison gives a
credible causal effect of roughly 1,700 dollars.

LaLonde's contribution was to replace the experiment's control group with an
unrelated population survey and ask whether statistical adjustment could recover
the experimental answer from that observational comparison. It largely could not:
the naive comparison suggests the programme reduced earnings by thousands of
dollars, because the survey population was far better off to begin with.

The dataset is included because it is a case where the true answer is known from
an experiment and the observational estimate is famously, catastrophically wrong.
Any honest account of what these methods deliver has to include it.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

#: Effect from the randomised comparison, in 1978 dollars. This is the number an
#: observational estimate is trying to recover.
EXPERIMENTAL_ATE = 1794.0

#: Columns recorded for every participant.
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

_DEHEJIA_WAHBA_BASE = "https://users.nber.org/~rdehejia/data"
_TREATED_FILE = "nsw_treated.txt"
_CONTROL_FILE = "nsw_control.txt"
_PSID_FILE = "psid_controls.txt"

_COLUMN_NAMES = [
    "treatment",
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re75",
    "re78",
]


def _fetch(filename: str, timeout: int = 20) -> pd.DataFrame | None:
    """Download one whitespace-delimited file from the reference archive.

    Returns ``None`` when the file cannot be retrieved, so a missing network
    connection degrades to the synthetic fallback rather than raising.
    """
    url = f"{_DEHEJIA_WAHBA_BASE}/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

    frame = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None)
    # The experimental extracts omit 1974 earnings, so the column count varies.
    if frame.shape[1] == len(_COLUMN_NAMES):
        frame.columns = _COLUMN_NAMES
        frame["re74"] = np.nan
    elif frame.shape[1] == len(_COLUMN_NAMES) + 1:
        frame.columns = _COLUMN_NAMES[:-2] + ["re74", "re75", "re78"]
    else:
        return None
    return frame


def _synthetic_fallback(seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Reproduce the shape of the benchmark when the archive is unreachable.

    The generated data mimics the structural problem that makes LaLonde hard: the
    comparison group is drawn from a much better-off population, so the naive
    contrast has the wrong sign. It is a stand-in for the real file, and is
    labelled as such in the returned ground truth.
    """
    rng = np.random.default_rng(seed)

    n_treated, n_control = 185, 2490
    # Programme participants were disadvantaged; the survey comparison group was not.
    treated_earnings_74 = rng.gamma(1.2, 1500, n_treated)
    control_earnings_74 = rng.gamma(4.0, 4200, n_control)

    treated = pd.DataFrame(
        {
            "treatment": 1,
            "age": rng.normal(25, 7, n_treated).clip(17, 55).round(),
            "education": rng.normal(10.3, 2.0, n_treated).clip(3, 16).round(),
            "black": rng.binomial(1, 0.84, n_treated),
            "hispanic": rng.binomial(1, 0.06, n_treated),
            "married": rng.binomial(1, 0.19, n_treated),
            "nodegree": rng.binomial(1, 0.71, n_treated),
            "re74": treated_earnings_74,
            "re75": treated_earnings_74 * 0.9 + rng.normal(0, 800, n_treated).clip(0, None),
        }
    )
    control = pd.DataFrame(
        {
            "treatment": 0,
            "age": rng.normal(34, 10, n_control).clip(17, 55).round(),
            "education": rng.normal(12.1, 3.0, n_control).clip(3, 16).round(),
            "black": rng.binomial(1, 0.25, n_control),
            "hispanic": rng.binomial(1, 0.03, n_control),
            "married": rng.binomial(1, 0.87, n_control),
            "nodegree": rng.binomial(1, 0.31, n_control),
            "re74": control_earnings_74,
            "re75": control_earnings_74 * 0.95 + rng.normal(0, 1200, n_control).clip(0, None),
        }
    )

    frame = pd.concat([treated, control], ignore_index=True)
    frame["re78"] = (
        0.85 * frame["re75"]
        + 300 * frame["education"]
        + EXPERIMENTAL_ATE * frame["treatment"]
        + rng.normal(0, 3000, len(frame))
    ).clip(0, None)

    naive = float(
        frame.loc[frame.treatment == 1, "re78"].mean()
        - frame.loc[frame.treatment == 0, "re78"].mean()
    )
    return frame, {
        "true_ate": EXPERIMENTAL_ATE,
        "naive_estimate": naive,
        "bias": naive - EXPERIMENTAL_ATE,
        "treatment": "treatment",
        "outcome": "re78",
        "covariates": COVARIATES,
        "source": "synthetic stand-in (the reference archive was unreachable)",
        "is_real_data": False,
        "why": (
            "The comparison group comes from a far better-off population, so the "
            "raw earnings difference reflects who was in each group rather than "
            "any effect of the programme."
        ),
    }


def load_lalonde(observational: bool = True, allow_fallback: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load the job training benchmark.

    Args:
        observational: When true, pair the programme participants with the
            population survey comparison group, which is the hard version of the
            problem. When false, return the randomised experiment, where the
            naive difference is already unbiased.
        allow_fallback: Generate a structurally similar stand-in when the
            reference archive cannot be reached, rather than raising.

    Returns:
        Tuple of the data and its ground truth. The ground truth carries
        ``is_real_data`` so downstream reporting never presents the stand-in as
        the genuine benchmark.

    Raises:
        RuntimeError: If the archive is unreachable and ``allow_fallback`` is off.
    """
    treated = _fetch(_TREATED_FILE)
    comparison = _fetch(_PSID_FILE if observational else _CONTROL_FILE)

    if treated is None or comparison is None:
        if not allow_fallback:
            raise RuntimeError(
                "The LaLonde archive could not be reached and the synthetic "
                "fallback was disabled."
            )
        return _synthetic_fallback()

    frame = pd.concat([treated, comparison], ignore_index=True)
    frame["re74"] = frame["re74"].fillna(frame["re75"])

    naive = float(
        frame.loc[frame.treatment == 1, "re78"].mean()
        - frame.loc[frame.treatment == 0, "re78"].mean()
    )

    if observational:
        why = (
            "The comparison group is a general population survey rather than the "
            "experiment's control group. Those people were older, better educated, "
            "and already earning far more, so the raw difference suggests the "
            "programme destroyed earnings when the experiment shows it helped."
        )
    else:
        why = (
            "Participants were randomised, so the raw difference already estimates "
            "the causal effect and no adjustment is needed."
        )

    return frame, {
        "true_ate": EXPERIMENTAL_ATE,
        "naive_estimate": naive,
        "bias": naive - EXPERIMENTAL_ATE,
        "treatment": "treatment",
        "outcome": "re78",
        "covariates": COVARIATES,
        "source": "National Supported Work demonstration (Dehejia and Wahba extract)",
        "is_real_data": True,
        "observational": observational,
        "why": why,
    }
