"""Generate a synthetic panel dataset for rural vitalization analysis.

The script constructs a province-year panel covering 2013-2023 that
includes agricultural loan indicators, the 13 secondary indicators of the
rural vitalization evaluation system, and composite scores based on the
weighting scheme described in the accompanying analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


OUTPUT_PATH = Path("data/rural_vitalization_panel.csv")
RNG_SEED = 1_234
YEARS: List[int] = list(range(2013, 2024))
REGIONS: List[str] = [
    "北京",
    "天津",
    "河北",
    "山西",
    "内蒙古",
    "辽宁",
    "吉林",
    "黑龙江",
    "上海",
    "江苏",
    "浙江",
    "安徽",
    "福建",
    "江西",
    "山东",
    "河南",
    "湖北",
    "湖南",
    "广东",
    "广西",
    "海南",
    "重庆",
    "四川",
    "贵州",
    "云南",
    "西藏",
    "陕西",
    "甘肃",
    "青海",
    "宁夏",
    "新疆",
]


@dataclass(frozen=True)
class IndicatorGroup:
    weights: Dict[str, float]
    negative: Iterable[str] = ()


INDICATOR_GROUPS: Dict[str, IndicatorGroup] = {
    "industry": IndicatorGroup(
        weights={
            "grain_yield_per_ha": 0.33,
            "per_capita_agri_output": 0.34,
            "agri_machinery_power_per_1000ha": 0.33,
        }
    ),
    "ecology": IndicatorGroup(
        weights={
            "forest_cover_rate": 0.40,
            "fertilizer_use_intensity": 0.30,
            "sanitary_toilet_rate": 0.30,
        },
        negative=("fertilizer_use_intensity",),
    ),
    "culture": IndicatorGroup(
        weights={
            "cultural_spending_share": 0.30,
            "township_culture_stations_per_10k": 0.35,
            "avg_education_years_rural": 0.35,
        }
    ),
    "governance": IndicatorGroup(
        weights={
            "urban_rural_governance_index": 0.50,
            "village_governance_participation_rate": 0.50,
        }
    ),
    "affluence": IndicatorGroup(
        weights={
            "rural_disposable_income_per_capita": 0.40,
            "rural_engel_coefficient": 0.30,
            "cars_per_100_households": 0.30,
        },
        negative=("rural_engel_coefficient",),
    ),
}

FIRST_LEVEL_WEIGHTS: Dict[str, float] = {
    "industry_score": 0.25,
    "ecology_score": 0.20,
    "culture_score": 0.15,
    "governance_score": 0.20,
    "affluence_score": 0.20,
}


def _baseline_draw(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def build_panel() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    records: List[Dict[str, float]] = []

    for region in REGIONS:
        base = {
            "grain": _baseline_draw(rng, 4.5, 7.5),
            "output": _baseline_draw(rng, 0.8, 3.0),
            "machinery": _baseline_draw(rng, 120, 420),
            "forest": _baseline_draw(rng, 15, 60),
            "fertilizer": _baseline_draw(rng, 0.35, 0.9),
            "toilet": _baseline_draw(rng, 55, 85),
            "cultural": _baseline_draw(rng, 6, 12),
            "station": _baseline_draw(rng, 0.9, 2.5),
            "education": _baseline_draw(rng, 7.0, 9.5),
            "governance": _baseline_draw(rng, 45, 75),
            "participation": _baseline_draw(rng, 40, 75),
            "income": _baseline_draw(rng, 9_000, 18_000),
            "engel": _baseline_draw(rng, 32, 42),
            "cars": _baseline_draw(rng, 8, 30),
            "loan": _baseline_draw(rng, 250, 850),
        }

        for year in YEARS:
            t = year - YEARS[0]
            trend = {
                "mild": 1 + 0.015 * t,
                "standard": 1 + 0.020 * t,
                "strong": 1 + 0.035 * t,
            }

            def jitter(scale: float) -> float:
                return float(rng.normal(0.0, scale))

            grain = base["grain"] * trend["standard"] + jitter(0.15)
            output = base["output"] * trend["strong"] + jitter(0.08)
            machinery = base["machinery"] * trend["strong"] + jitter(12)
            forest = np.clip(base["forest"] * trend["mild"] + jitter(1.8), 12, 80)
            fertilizer = np.clip(
                base["fertilizer"] * (1 - 0.01 * t) + jitter(0.015), 0.2, 1.2
            )
            toilet = np.clip(base["toilet"] * trend["strong"] + jitter(1.5), 40, 100)
            cultural = np.clip(base["cultural"] * trend["mild"] + jitter(0.25), 4.5, 18)
            station = np.clip(base["station"] * trend["standard"] + jitter(0.08), 0.5, 4)
            education = np.clip(base["education"] + 0.05 * t + jitter(0.05), 6.5, 12)
            governance = np.clip(base["governance"] * trend["standard"] + jitter(1.2), 40, 95)
            participation = np.clip(
                base["participation"] * trend["standard"] + jitter(1.5), 30, 90
            )
            income = base["income"] * trend["strong"] * (1 + 0.01 * jitter(1))
            engel = np.clip(base["engel"] * (1 - 0.02 * t) + jitter(0.4), 20, 45)
            cars = np.clip(base["cars"] * trend["strong"] + jitter(1.5), 5, 90)
            loan = base["loan"] * trend["strong"] * (1 + 0.015 * t) + jitter(15)

            records.append(
                {
                    "year": year,
                    "region": region,
                    "agri_loan_balance": round(loan, 2),
                    "grain_yield_per_ha": round(grain, 3),
                    "per_capita_agri_output": round(output, 3),
                    "agri_machinery_power_per_1000ha": round(machinery, 2),
                    "forest_cover_rate": round(forest, 2),
                    "fertilizer_use_intensity": round(fertilizer, 3),
                    "sanitary_toilet_rate": round(toilet, 2),
                    "cultural_spending_share": round(cultural, 3),
                    "township_culture_stations_per_10k": round(station, 3),
                    "avg_education_years_rural": round(education, 3),
                    "urban_rural_governance_index": round(governance, 2),
                    "village_governance_participation_rate": round(participation, 2),
                    "rural_disposable_income_per_capita": round(income, 2),
                    "rural_engel_coefficient": round(engel, 2),
                    "cars_per_100_households": round(cars, 2),
                }
            )

    df = pd.DataFrame(records).sort_values(["region", "year"], ignore_index=True)
    df["agri_loan_growth_rate"] = (
        df.groupby("region")["agri_loan_balance"].pct_change().fillna(0.0)
    )

    normalized: Dict[str, pd.Series] = {}
    for group in INDICATOR_GROUPS.values():
        for indicator in group.weights:
            if indicator in normalized:
                continue
            col = df[indicator]
            span = col.max() - col.min()
            if span == 0:
                norm = pd.Series(1.0, index=df.index)
            else:
                norm = (col - col.min()) / span
            if indicator in group.negative:
                norm = 1 - norm
            normalized[indicator] = norm

    for name, group in INDICATOR_GROUPS.items():
        score = sum(group.weights[ind] * normalized[ind] for ind in group.weights)
        score_min = score.min()
        score_max = score.max()
        if score_max > score_min:
            score = (score - score_min) / (score_max - score_min)
        df[f"{name}_score"] = 100 * score

    composite = sum(
        FIRST_LEVEL_WEIGHTS[col] * df[col] for col in FIRST_LEVEL_WEIGHTS
    )
    composite_min = composite.min()
    composite_max = composite.max()
    if composite_max > composite_min:
        composite = 100 * (composite - composite_min) / (composite_max - composite_min)
    else:
        composite = pd.Series(100, index=df.index)
    df["rural_vitalization_index"] = composite

    df["post_rural_vitalization"] = (df["year"] >= 2018).astype(int)

    output_columns = [
        "year",
        "region",
        "post_rural_vitalization",
        "agri_loan_balance",
        "agri_loan_growth_rate",
        "grain_yield_per_ha",
        "per_capita_agri_output",
        "agri_machinery_power_per_1000ha",
        "forest_cover_rate",
        "fertilizer_use_intensity",
        "sanitary_toilet_rate",
        "cultural_spending_share",
        "township_culture_stations_per_10k",
        "avg_education_years_rural",
        "urban_rural_governance_index",
        "village_governance_participation_rate",
        "rural_disposable_income_per_capita",
        "rural_engel_coefficient",
        "cars_per_100_households",
        "industry_score",
        "ecology_score",
        "culture_score",
        "governance_score",
        "affluence_score",
        "rural_vitalization_index",
    ]

    return df[output_columns]


def main() -> None:
    df = build_panel()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
