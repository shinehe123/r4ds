# R for Data Science

## Listed-company city innovation network

`city_innovation_network.py` separates company-to-city mapping from aggregation
of real company-level patent collaboration/citation links.

```bash
python -m pip install -r requirements.txt
python city_innovation_network.py map-companies \
  --companies companies_上市公司自身_with_address.csv \
  --output outputs/company_city_crosswalk.csv \
  --audit-output outputs/company_city_unresolved.csv

python city_innovation_network.py build-network \
  --companies companies_上市公司自身_with_address.csv \
  --links patent_collaborations.csv \
  --output-dir outputs
```

For the preferred patent co-application specification (one row per
listed-company/partner/patent), use the undirected builder and the Repaco
intra-group exclusion table:

```bash
python city_innovation_network.py build-coapplication \
  --companies companies_上市公司自身_with_address.csv \
  --links patent_collaborations.csv \
  --repaco repaco_members_all.csv \
  --output-dir outputs/coapplication
```

Required link concepts are: patent ID, application year/date, listed-company
code/name, partner code/name, and (for a non-listed partner) partner address.
Repeated API rows are deduplicated at patent--city-pair level. Each distinct
joint patent contributes weight 1 to an unordered city pair.

Common English/Chinese columns are auto-detected; explicit `--*-col` options
cover other schemas. Use `--directed` for citation flows. Co-application
networks are undirected by default. Same-city links are excluded by default and
retained in `company_link_audit.csv`.

`repaco_members_all.csv` is a corporate-group membership table, not an
innovation edge list: it lacks patent IDs, partner addresses and event years.
Use it to flag/exclude intra-group relations upstream, not as city network
edges.
[![Travis build status](https://travis-ci.org/hadley/r4ds.svg?branch=master)](https://travis-ci.org/hadley/r4ds)

This repository contains the source of [R for Data Science](http://r4ds.had.co.nz)
book. The book is built using [bookdown](https://github.com/rstudio/bookdown).

The R packages used in this book can be installed via

```{r}
devtools::install_github("hadley/r4ds")
```
