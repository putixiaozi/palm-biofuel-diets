#!/bin/zsh
# Full rebuild. Steps 01-03 need FABIO version 2 (see data_fabio/README.md); all later steps read data_derived/.
set -e
cd "$(dirname "$0")"
mkdir -p logs results data_derived figures
run() { echo "== $1"; shift; "$@" > "logs/$(basename $2 .py).log" 2>&1; }
Rscript code/01_extract_fabio.R
run fbs python3 code/02_extract_fbs.py
run base python3 code/03_build_base.py
for s in 04_history 05_experiments 09_robustness 10_layers_m0 08_analysis; do run $s python3 code/$s.py; done
run 07_kink python3 code/07_kink.py 7
python3 code/07_kink.py 7 pcap > logs/07_kink_pcap.log 2>&1
run 06_country python3 code/06_country.py 6
for s in 11_country_analysis 20_relocation_options 16_revision_checks 19_fair_country_allocations 18_reference_diagnostic \
         17_feedstock_validation 12_si_data 23_food_basis_headline 24_land_accounting 25_allocation_rules 21_key_numbers \
         22_supplementary_data; do run $s python3 code/$s.py; done
for s in fig0 fig1 fig2 fig3 fig4 fig5 figS figS14; do run $s python3 code/$s.py; done
echo done
