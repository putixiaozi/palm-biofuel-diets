"""Generate the four efficient-and-fair country allocation scenarios used in
Supplementary Data 3.

This is a focused runner for the fair-country table. It applies the same model
settings as the corresponding scenarios in 16_revision_checks.py while avoiding
the unrelated frontier and boundary searches.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import model as M


ROOT = Path(M.ROOT)
OUT = ROOT / "results"
BASE = M.System(M.load_year(2023))
FUTURE = M.project(BASE)

SCENARIOS = (
    ("baseline_fair", {}),
    ("relocation_0.25_fair", {"relocation": 0.25}),
    ("relocation_0.5_fair", {"relocation": 0.5}),
    ("palm_full_fair", {"full_palm": True}),
    ("pcap_alt_fair", {"pcap": True}),
)


def evaluate(name, m, relocation=0.0, palm_share=0.0, pcap=False, full_palm=False):
    system = FUTURE.copy()
    targets = system.targets("eat_quality", palm_cap=pcap)
    links, import_caps = M.build_links(system)
    fuel_cap = system.fuel_base * m * (1 + relocation)
    fuel_lower = system.fuel_base * m * (1 - relocation) if relocation else None
    eligible = system.eligibility(policy=palm_share == 0 and not full_palm)

    palm_fuel_cap = None
    if palm_share:
        palm_fuel_cap = np.full(len(system.C), np.inf)
        for country in M.NO_PALM_FUEL:
            if country in system.ci:
                index = system.ci[country]
                palm_fuel_cap[index] = palm_share * system.fuel_base[index] * m

    result = M.solve(
        system,
        targets,
        links,
        import_caps,
        fuel_cap,
        eligible,
        fuel_min=m * system.fuel_base.sum() * (1 - 1e-7),
        fuel_lower=fuel_lower,
        palm_fuel_cap=palm_fuel_cap,
        mode="waterfill",
        gap_tol=1e-4,
        detail=True,
    )
    assert result["feasible"], (name, m)
    if relocation and result["fuel_unmet"] < 1:
        delivered = result["B"].sum(axis=1) / M.UNIT
        assert np.max(delivered - fuel_cap) < 1e-4
        assert np.min(delivered - fuel_lower) > -1e-4
    if palm_fuel_cap is not None:
        assert np.max(result["B"][:, system.palm] / M.UNIT - palm_fuel_cap) < 1e-4

    row = {
        "scenario": name,
        "year": 2035,
        "target": "eat_quality",
        "m": m,
        "requested_Mt": m * system.fuel_base.sum() / 1000,
        "delivered_Mt": result["fuel"] / 1e6,
        "unmet_Mt": result["fuel_unmet"] / 1e6,
        "cut_Mt": result["cut"] / 1e6,
        "gap_Mt": result["gap"] / 1e6,
        "relocation": relocation,
        "palm_share": palm_share,
        "uco": 0.0,
        "supply_scale": 1.0,
    }
    countries = pd.DataFrame(
        {
            "scenario": name,
            "m": m,
            "iso3c": system.C,
            "shortfall_Mt": np.where(
                np.abs(result["u"]) < 1e-6, 0.0, result["u"]
            )
            / 1e6,
            "population": system.pop,
            "target_Mt": targets["D"] / 1000,
        }
    )
    return row, countries


def main():
    rows = []
    countries = []
    for name, settings in SCENARIOS:
        for m in (0.0, 1.0):
            row, country = evaluate(name, m, **settings)
            rows.append(row)
            countries.append(country)
            print(row, flush=True)

    pd.DataFrame(rows).to_csv(OUT / "revision_fair_scenarios.csv", index=False)
    pd.concat(countries, ignore_index=True).to_csv(
        OUT / "revision_country_allocations.csv", index=False
    )
    print("FAIR COUNTRY ALLOCATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
