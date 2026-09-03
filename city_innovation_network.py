#!/usr/bin/env python3
"""Map listed companies to Chinese prefectures and aggregate innovation links.

``map-companies`` parses registered/office addresses into an auditable
company--city crosswalk. ``build-network`` joins a company-level patent
collaboration/citation edge list to that crosswalk and aggregates annual
inter-city networks.

Repaco group-member records are not interpreted as innovation links. They are
useful as an exclusion list but contain neither patent identifiers nor member
locations and event years.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import pandas as pd


CODE_ALIASES = ("code", "股票代码", "证券代码", "stock_code")
NAME_ALIASES = ("name", "公司名称", "公司中文名称", "企业名称")
SHORT_NAME_ALIASES = ("aliases", "证券简称", "证券名称", "简称")
ADDRESS_ALIASES = ("address", "注册地址", "办公地址", "公司地址")
SOURCE_CODE_ALIASES = (
    "source_code", "applicant_code", "citing_code", "引用方代码", "申请人代码",
)
TARGET_CODE_ALIASES = (
    "target_code", "partner_code", "cited_code", "被引用方代码", "合作方代码",
)
SOURCE_NAME_ALIASES = (
    "source_name", "applicant_name", "citing_name", "引用方", "申请人",
)
TARGET_NAME_ALIASES = (
    "target_name", "partner_name", "cited_name", "被引用方", "合作方",
)
YEAR_ALIASES = ("year", "申请年", "申请年份", "被引用年度", "年份")
WEIGHT_ALIASES = (
    "weight", "count", "citation_count", "被引用次数", "合作专利数", "专利数",
)
PATENT_ALIASES = ("patent_id", "申请号", "专利申请号", "publication_number")
SOURCE_ADDRESS_ALIASES = ("source_address", "listed_address", "上市公司地址", "申请人地址")
TARGET_ADDRESS_ALIASES = ("target_address", "partner_address", "合作方地址", "共同申请人地址")
DATE_ALIASES = ("application_date", "申请日", "申请日期")
MUNICIPALITIES = {"北京市", "天津市", "上海市", "重庆市"}
SUFFIXES = (
    "特别行政区", "维吾尔自治区", "壮族自治区", "回族自治区", "自治区",
    "自治州", "地区", "盟", "省", "市",
)
COUNTY_SUFFIXES = ("自治县", "自治旗", "新区", "矿区", "林区", "特区", "县", "市", "区", "旗")


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV/Excel while preserving leading zeros in security codes."""
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, dtype=str, low_memory=False)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def choose_column(
    frame: pd.DataFrame,
    explicit: Optional[str],
    aliases: Iterable[str],
    label: str,
    required: bool = True,
) -> Optional[str]:
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"Column '{explicit}' supplied for {label} was not found")
        return explicit
    for candidate in aliases:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(f"Could not identify {label}; available columns: {list(frame.columns)}")
    return None


def normalize_code(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = re.sub(r"\.0+$", "", str(value).strip())
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def normalize_name(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = re.sub(r"[\s（）()·•]", "", str(value)).upper()
    text = re.sub(
        r"(?:集团股份有限公司|股份有限公司|有限责任公司|集团有限公司|有限公司|股份公司|公司)$",
        "",
        text,
    )
    return text or None


def strip_admin_suffix(name: str) -> str:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def strip_county_suffix(name: str) -> str:
    for suffix in COUNTY_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def default_adcodes_path() -> Path:
    """Locate cpca's adcode table without importing its legacy package API."""
    try:
        dist = importlib.metadata.distribution("cpca")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Administrative divisions unavailable. Run: pip install -r requirements.txt"
        ) from exc
    path = Path(dist.locate_file("cpca/resources/adcodes.csv"))
    if not path.exists():
        raise RuntimeError(f"Administrative-code table not found: {path}")
    return path


@dataclass(frozen=True)
class AdminUnit:
    adcode: str
    name: str
    rank: int
    province_code: str
    city_code: Optional[str]


class AddressMapper:
    """Deterministic address-to-prefecture mapper backed by PRC adcodes."""

    def __init__(self, adcodes_path: Path) -> None:
        raw = pd.read_csv(adcodes_path, dtype=str)
        if not {"adcode", "name"}.issubset(raw.columns):
            raise ValueError("adcodes file must contain adcode and name columns")
        self.units: list[AdminUnit] = []
        self.by_code: dict[str, AdminUnit] = {}
        for row in raw[["adcode", "name"]].dropna().itertuples(index=False):
            code = str(row.adcode)[:6]
            rank = 0 if code.endswith("0000") else 1 if code.endswith("00") else 2
            unit = AdminUnit(
                code, str(row.name), rank, code[:2] + "0000",
                (code[:4] + "00") if rank >= 1 else None,
            )
            self.units.append(unit)
            self.by_code[code] = unit
        self.provinces = [u for u in self.units if u.rank == 0]
        # ``市辖区``/``县`` are placeholder second-level names used beneath
        # municipalities. Matching those literal words in a street address
        # would create false city assignments.
        self.cities = [
            u for u in self.units if u.rank == 1 and u.name not in {"市辖区", "县"}
        ]
        self.counties = [u for u in self.units if u.rank == 2]

    @staticmethod
    def _matches(text: str, unit: AdminUnit, allow_short: bool) -> bool:
        if unit.name in text:
            return True
        short = strip_admin_suffix(unit.name)
        return allow_short and len(short) >= 2 and short in text

    @staticmethod
    def _position(text: str, unit: AdminUnit) -> int:
        positions = [
            p for p in (text.find(unit.name), text.find(strip_admin_suffix(unit.name))) if p >= 0
        ]
        return min(positions) if positions else 10**9

    def _province(self, text: str) -> Optional[AdminUnit]:
        # A full province name is strong evidence anywhere in the address.
        # A stripped alias (e.g. ``山东``) is only safe near the beginning;
        # otherwise street/community names such as ``松坪山东物`` create
        # spurious province hints that suppress an explicit city match.
        full_hits = [u for u in self.provinces if u.name in text]
        if full_hits:
            return min(full_hits, key=lambda u: text.find(u.name))
        short_hits = [
            u for u in self.provinces
            if 0 <= text.find(strip_admin_suffix(u.name)) <= 8
        ]
        return min(short_hits, key=lambda u: self._position(text, u)) if short_hits else None

    def map_one(self, address: object) -> dict[str, object]:
        if pd.isna(address) or not str(address).strip():
            return self._empty("missing_address")
        text = re.sub(r"\s+", "", str(address))
        province_hint = self._province(text)
        city_hits = [u for u in self.cities if self._matches(text, u, True)]
        # An explicit/leading city is stronger than a stripped province word
        # embedded in a road name (e.g. 福州市...儒江西路 or 北海市西藏路).
        if province_hint and province_hint.name not in text:
            strong_city_hits = [
                u for u in city_hits
                if u.name in text or self._position(text, u) == 0
            ]
            if strong_city_hits and not any(
                u.province_code == province_hint.adcode for u in strong_city_hits
            ):
                province_hint = None
        if province_hint:
            city_hits = [u for u in city_hits if u.province_code == province_hint.adcode]
            province_short = strip_admin_suffix(province_hint.name)
            city_hits = [
                u for u in city_hits
                if u.name in text
                or strip_admin_suffix(u.name) != province_short
                or text.find(province_short, len(province_hint.name)) >= 0
            ]
        county_hits = [u for u in self.counties if self._matches(text, u, False)]
        if province_hint:
            county_hits = [u for u in county_hits if u.province_code == province_hint.adcode]
        county_alias_match = False
        if not county_hits:
            # Industrial-zone addresses frequently omit 县/区/市 from a
            # county-level name (``江苏省丹阳经济开发区``). A province hint, or
            # a unique county alias at position zero, makes that inference
            # sufficiently constrained while remaining auditable as medium
            # confidence.
            alias_hits = [
                u for u in self.counties
                if len(strip_county_suffix(u.name)) >= 2
                and strip_county_suffix(u.name) in text
            ]
            if province_hint:
                alias_hits = [u for u in alias_hits if u.province_code == province_hint.adcode]
            else:
                alias_hits = [
                    u for u in alias_hits if text.startswith(strip_county_suffix(u.name))
                ]
            if len({u.city_code for u in alias_hits}) == 1:
                county_hits = alias_hits
                county_alias_match = bool(alias_hits)

        city: Optional[AdminUnit] = None
        county: Optional[AdminUnit] = None
        method, confidence = "unresolved", "unresolved"
        if city_hits:
            # Prefer an explicit full administrative name (``深圳市``) over a
            # shorter alias that happens to occur earlier in the address.
            city = min(city_hits, key=lambda u: (0 if u.name in text else 1, self._position(text, u)))
            compatible_counties = [u for u in county_hits if u.city_code == city.adcode]
            if compatible_counties:
                county = min(compatible_counties, key=lambda u: self._position(text, u))
            explicit = city.name in text
            method, confidence = ("explicit_city", "high") if explicit else ("city_alias", "medium")
        elif county_hits:
            parent_codes = {u.city_code for u in county_hits}
            if len(parent_codes) == 1:
                county = min(county_hits, key=lambda u: self._position(text, u))
                city = self.by_code.get(county.city_code or "")
                method = "county_alias_inference" if county_alias_match else "county_inference"
                confidence = "medium" if county_alias_match else ("high" if province_hint else "medium")
        elif province_hint and province_hint.name in MUNICIPALITIES:
            method, confidence = "municipality", "high"

        province = province_hint
        if city and not province:
            province = self.by_code.get(city.province_code)
        if county and not province:
            province = self.by_code.get(county.province_code)
        if province and province.name in MUNICIPALITIES:
            city_name, city_code = province.name, province.adcode
        elif city:
            city_name, city_code = city.name, city.adcode
        else:
            city_name, city_code = None, None
        return {
            "province": province.name if province else None,
            "province_code": province.adcode if province else None,
            "city": city_name, "city_code": city_code,
            "county": county.name if county else None,
            "county_code": county.adcode if county else None,
            "match_method": method, "match_confidence": confidence,
        }

    @staticmethod
    def _empty(method: str) -> dict[str, object]:
        return {
            "province": None, "province_code": None, "city": None, "city_code": None,
            "county": None, "county_code": None, "match_method": method,
            "match_confidence": "unresolved",
        }


def map_companies(
    companies: pd.DataFrame,
    mapper: AddressMapper,
    code_col: Optional[str] = None,
    name_col: Optional[str] = None,
    alias_col: Optional[str] = None,
    address_col: Optional[str] = None,
) -> pd.DataFrame:
    code_col = choose_column(companies, code_col, CODE_ALIASES, "company code")
    name_col = choose_column(companies, name_col, NAME_ALIASES, "company name")
    alias_col = choose_column(companies, alias_col, SHORT_NAME_ALIASES, "company alias", False)
    address_col = choose_column(companies, address_col, ADDRESS_ALIASES, "company address")
    result = pd.DataFrame({
        "company_code": companies[code_col].map(normalize_code),
        "company_name": companies[name_col],
        "company_alias": companies[alias_col] if alias_col else None,
        "address": companies[address_col],
    })
    mapped = pd.DataFrame([mapper.map_one(v) for v in result.address], index=result.index)
    result = pd.concat([result, mapped], axis=1)
    result["city_id"] = result.city_code
    return result


def build_company_lookup(crosswalk: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    code_lookup = {
        code: city for code, city in zip(crosswalk.company_code, crosswalk.city_code)
        if pd.notna(code) and pd.notna(city)
    }
    name_lookup: dict[str, str] = {}
    ambiguous: set[str] = set()
    for row in crosswalk.itertuples(index=False):
        if pd.isna(row.city_code):
            continue
        for value in (row.company_name, row.company_alias):
            key = normalize_name(value)
            if not key:
                continue
            if key in name_lookup and name_lookup[key] != row.city_code:
                ambiguous.add(key)
            else:
                name_lookup[key] = row.city_code
    for key in ambiguous:
        name_lookup.pop(key, None)
    return code_lookup, name_lookup


def prepare_company_edges(
    links: pd.DataFrame, crosswalk: pd.DataFrame, args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_code_col = choose_column(links, args.source_code_col, SOURCE_CODE_ALIASES, "source code", False)
    target_code_col = choose_column(links, args.target_code_col, TARGET_CODE_ALIASES, "target code", False)
    source_name_col = choose_column(links, args.source_name_col, SOURCE_NAME_ALIASES, "source name", False)
    target_name_col = choose_column(links, args.target_name_col, TARGET_NAME_ALIASES, "target name", False)
    if not (source_code_col or source_name_col) or not (target_code_col or target_name_col):
        raise ValueError("Links need source and target identifiers (codes and/or names)")
    year_col = choose_column(links, args.year_col, YEAR_ALIASES, "year")
    weight_col = choose_column(links, args.weight_col, WEIGHT_ALIASES, "weight", False)
    patent_col = choose_column(links, args.patent_col, PATENT_ALIASES, "patent id", False)
    code_lookup, name_lookup = build_company_lookup(crosswalk)
    work = links.copy()
    work["source_code"] = work[source_code_col].map(normalize_code) if source_code_col else None
    work["target_code"] = work[target_code_col].map(normalize_code) if target_code_col else None
    work["source_name"] = work[source_name_col] if source_name_col else None
    work["target_name"] = work[target_name_col] if target_name_col else None
    work["year"] = pd.to_numeric(work[year_col], errors="coerce").astype("Int64")
    work["weight"] = pd.to_numeric(work[weight_col], errors="coerce").fillna(1.0) if weight_col else 1.0
    work["patent_id"] = work[patent_col] if patent_col else None

    def resolve(code: object, name: object) -> Optional[str]:
        normalized_code = normalize_code(code)
        if normalized_code in code_lookup:
            return code_lookup[normalized_code]
        normalized_name = normalize_name(name)
        return name_lookup.get(normalized_name) if normalized_name else None

    work["source_city_code"] = [resolve(c, n) for c, n in zip(work.source_code, work.source_name)]
    work["target_city_code"] = [resolve(c, n) for c, n in zip(work.target_code, work.target_name)]

    def reason(row: pd.Series) -> str:
        if pd.isna(row.year):
            return "missing_year"
        if pd.isna(row.source_city_code):
            return "unmapped_source"
        if pd.isna(row.target_city_code):
            return "unmapped_target"
        if row.source_city_code == row.target_city_code and not args.include_intra_city:
            return "intra_city"
        return "included"

    audit = work.copy()
    audit["exclusion_reason"] = audit.apply(reason, axis=1)
    return work.loc[audit.exclusion_reason == "included"].copy(), audit


def build_repaco_pairs(repaco: Optional[pd.DataFrame]) -> set[tuple[str, str]]:
    """Return normalized (listed code, group-member name) pairs."""
    if repaco is None:
        return set()
    code_col = choose_column(repaco, None, CODE_ALIASES, "Repaco listed-company code")
    name_col = choose_column(repaco, None, NAME_ALIASES, "Repaco member name")
    return {
        (code, name)
        for code, name in zip(repaco[code_col].map(normalize_code), repaco[name_col].map(normalize_name))
        if code and name
    }


def prepare_coapplication_edges(
    links: pd.DataFrame,
    crosswalk: pd.DataFrame,
    mapper: AddressMapper,
    repaco: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare one-listed-company/one-partner/one-patent records.

    Each patent-city pair is deduplicated before aggregation, so repeated API
    rows or multiple applicants from the same two cities cannot inflate weight.
    """
    source_code_col = choose_column(
        links, args.source_code_col, SOURCE_CODE_ALIASES + CODE_ALIASES, "listed-company code", False
    )
    target_code_col = choose_column(
        links, args.target_code_col, TARGET_CODE_ALIASES, "partner code", False
    )
    source_name_col = choose_column(
        links, args.source_name_col, SOURCE_NAME_ALIASES + NAME_ALIASES, "listed-company name", False
    )
    target_name_col = choose_column(
        links, args.target_name_col, TARGET_NAME_ALIASES, "partner name", False
    )
    target_address_col = choose_column(
        links, args.target_address_col, TARGET_ADDRESS_ALIASES, "partner address", False
    )
    patent_col = choose_column(links, args.patent_col, PATENT_ALIASES, "patent id")
    year_col = choose_column(links, args.year_col, YEAR_ALIASES, "application year", False)
    date_col = choose_column(links, args.date_col, DATE_ALIASES, "application date", False)
    if not (source_code_col or source_name_col):
        raise ValueError("Co-application data need a listed-company code or name")
    if not (target_code_col or target_name_col):
        raise ValueError("Co-application data need a partner code or name")
    if not (year_col or date_col):
        raise ValueError("Co-application data need an application year or date")

    code_lookup, name_lookup = build_company_lookup(crosswalk)
    city_by_code = crosswalk.drop_duplicates("city_code").set_index("city_code")["city"].to_dict()
    group_pairs = build_repaco_pairs(repaco)
    work = links.copy()
    work["source_code"] = work[source_code_col].map(normalize_code) if source_code_col else None
    work["target_code"] = work[target_code_col].map(normalize_code) if target_code_col else None
    work["source_name"] = work[source_name_col] if source_name_col else None
    work["target_name"] = work[target_name_col] if target_name_col else None
    work["target_address"] = work[target_address_col] if target_address_col else None
    work["patent_id"] = work[patent_col].astype("string").str.strip()
    if year_col:
        work["year"] = pd.to_numeric(work[year_col], errors="coerce").astype("Int64")
    else:
        work["year"] = pd.to_datetime(work[date_col], errors="coerce").dt.year.astype("Int64")

    def lookup_city(code: object, name: object) -> Optional[str]:
        normalized_code = normalize_code(code)
        if normalized_code and normalized_code in code_lookup:
            return code_lookup[normalized_code]
        normalized_name = normalize_name(name)
        return name_lookup.get(normalized_name) if normalized_name else None

    work["source_city_code"] = [
        lookup_city(c, n) for c, n in zip(work.source_code, work.source_name)
    ]
    target_city_codes: list[Optional[str]] = []
    target_location_methods: list[str] = []
    for code, name, address in zip(work.target_code, work.target_name, work.target_address):
        city_code = lookup_city(code, name)
        if city_code:
            target_city_codes.append(city_code)
            target_location_methods.append("listed_company_lookup")
        elif pd.notna(address) and str(address).strip():
            mapped = mapper.map_one(address)
            target_city_codes.append(mapped["city_code"])
            target_location_methods.append(f"address:{mapped['match_method']}")
        else:
            target_city_codes.append(None)
            target_location_methods.append("unresolved")
    work["target_city_code"] = target_city_codes
    work["target_location_method"] = target_location_methods
    work["same_group"] = [
        (normalize_code(code), normalize_name(name)) in group_pairs
        for code, name in zip(work.source_code, work.target_name)
    ]

    def reason(row: pd.Series) -> str:
        if pd.isna(row.patent_id) or not str(row.patent_id).strip():
            return "missing_patent_id"
        if pd.isna(row.year):
            return "missing_year"
        if pd.isna(row.source_city_code):
            return "unmapped_listed_company"
        if row.same_group:
            return "same_corporate_group"
        if pd.isna(row.target_city_code):
            return "unmapped_partner_city"
        if row.source_city_code == row.target_city_code and not args.include_intra_city:
            return "intra_city"
        return "included"

    audit = work.copy()
    audit["exclusion_reason"] = audit.apply(reason, axis=1)
    valid = audit[audit.exclusion_reason == "included"].copy()
    if not valid.empty:
        canonical = valid[["source_city_code", "target_city_code"]].apply(
            lambda r: sorted((r.iloc[0], r.iloc[1])), axis=1, result_type="expand"
        )
        valid[["source_city_code", "target_city_code"]] = canonical
        valid["source_city"] = valid.source_city_code.map(city_by_code)
        valid["target_city"] = valid.target_city_code.map(city_by_code)
        valid = valid.drop_duplicates(
            ["patent_id", "year", "source_city_code", "target_city_code"]
        )
        valid["weight"] = 1.0
    return valid, audit


def aggregate_city_network(
    company_edges: pd.DataFrame, crosswalk: pd.DataFrame, directed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edges = company_edges.copy()
    if not directed:
        pairs = edges[["source_city_code", "target_city_code"]].apply(
            lambda r: sorted((r.iloc[0], r.iloc[1])), axis=1, result_type="expand"
        )
        edges[["source_city_code", "target_city_code"]] = pairs
    city_edges = edges.groupby(
        ["year", "source_city_code", "target_city_code"], as_index=False
    ).agg(weight=("weight", "sum"), company_link_records=("weight", "size"))
    names = (
        crosswalk.dropna(subset=["city_code"]).drop_duplicates("city_code")
        .set_index("city_code")[["province", "city"]]
    )
    city_edges["source_city"] = city_edges.source_city_code.map(names.city)
    city_edges["target_city"] = city_edges.target_city_code.map(names.city)
    city_edges["source_province"] = city_edges.source_city_code.map(names.province)
    city_edges["target_province"] = city_edges.target_city_code.map(names.province)
    node_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    graph_type = nx.DiGraph if directed else nx.Graph
    for year, year_edges in city_edges.groupby("year", sort=True):
        graph = graph_type()
        for row in year_edges.itertuples(index=False):
            graph.add_edge(row.source_city_code, row.target_city_code, weight=float(row.weight))
        degree = dict(graph.degree())
        strength = dict(graph.degree(weight="weight"))
        betweenness = nx.betweenness_centrality(graph, weight=None, normalized=True)
        pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_nodes() else {}
        clustering = nx.clustering(graph.to_undirected(), weight="weight")
        for code in graph.nodes:
            node_rows.append({
                "year": int(year), "city_code": code,
                "province": names.at[code, "province"], "city": names.at[code, "city"],
                "degree": degree[code], "weighted_degree": strength[code],
                "betweenness": betweenness[code], "pagerank": pagerank[code],
                "clustering": clustering[code],
            })
        summary_rows.append({
            "year": int(year), "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(), "total_weight": float(year_edges.weight.sum()),
            "density": nx.density(graph),
            "components": nx.number_weakly_connected_components(graph)
            if directed else nx.number_connected_components(graph),
        })
    return city_edges, pd.DataFrame(node_rows), pd.DataFrame(summary_rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def command_map(args: argparse.Namespace) -> None:
    mapper = AddressMapper(args.adcodes or default_adcodes_path())
    crosswalk = map_companies(
        read_table(args.companies), mapper, args.code_col, args.name_col,
        args.alias_col, args.address_col,
    )
    write_csv(crosswalk, args.output)
    write_csv(crosswalk[crosswalk.city_code.isna()], args.audit_output)
    city_counts = (
        crosswalk.dropna(subset=["city_code"])
        .groupby(["province", "province_code", "city", "city_code"], as_index=False)
        .agg(
            listed_company_count=("company_code", "nunique"),
            high_confidence_count=(
                "match_confidence", lambda values: int((values == "high").sum())
            ),
            medium_confidence_count=(
                "match_confidence", lambda values: int((values == "medium").sum())
            ),
        )
        .sort_values("listed_company_count", ascending=False)
    )
    city_counts_path = args.output.with_name("city_listed_company_counts.csv")
    write_csv(city_counts, city_counts_path)
    mapped = crosswalk.city_code.notna().sum()
    print(f"Mapped {mapped:,}/{len(crosswalk):,} companies ({mapped / len(crosswalk):.2%})")
    print(f"Crosswalk: {args.output}")
    print(f"City counts: {city_counts_path}")
    print(f"Unresolved audit: {args.audit_output}")


def command_network(args: argparse.Namespace) -> None:
    mapper = AddressMapper(args.adcodes or default_adcodes_path())
    crosswalk = map_companies(
        read_table(args.companies), mapper, args.code_col, args.name_col,
        args.alias_col, args.address_col,
    )
    company_edges, audit = prepare_company_edges(read_table(args.links), crosswalk, args)
    if company_edges.empty:
        raise ValueError(
            f"No valid company links remain after mapping: {audit.exclusion_reason.value_counts().to_dict()}"
        )
    city_edges, city_nodes, evolution = aggregate_city_network(company_edges, crosswalk, args.directed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(crosswalk, args.output_dir / "company_city_crosswalk.csv")
    write_csv(audit, args.output_dir / "company_link_audit.csv")
    write_csv(city_edges, args.output_dir / "city_innovation_edges.csv")
    write_csv(city_nodes, args.output_dir / "city_innovation_nodes.csv")
    write_csv(evolution, args.output_dir / "city_network_evolution.csv")
    print(evolution.to_string(index=False))
    print(f"Outputs: {args.output_dir}")


def command_coapplication(args: argparse.Namespace) -> None:
    mapper = AddressMapper(args.adcodes or default_adcodes_path())
    crosswalk = map_companies(
        read_table(args.companies), mapper, args.code_col, args.name_col,
        args.alias_col, args.address_col,
    )
    repaco = read_table(args.repaco) if args.repaco else None
    patent_edges, audit = prepare_coapplication_edges(
        read_table(args.links), crosswalk, mapper, repaco, args
    )
    if patent_edges.empty:
        raise ValueError(
            "No valid inter-city co-application links remain: "
            f"{audit.exclusion_reason.value_counts().to_dict()}"
        )
    city_edges, city_nodes, evolution = aggregate_city_network(
        patent_edges, crosswalk, directed=False
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(crosswalk, args.output_dir / "company_city_crosswalk.csv")
    write_csv(audit, args.output_dir / "coapplication_record_audit.csv")
    write_csv(patent_edges, args.output_dir / "patent_city_pairs_deduplicated.csv")
    write_csv(city_edges, args.output_dir / "city_coapplication_edges.csv")
    write_csv(city_nodes, args.output_dir / "city_coapplication_nodes.csv")
    write_csv(evolution, args.output_dir / "city_coapplication_evolution.csv")
    print(evolution.to_string(index=False))
    print("Audit:", audit.exclusion_reason.value_counts().to_dict())
    print(f"Outputs: {args.output_dir}")


def add_mapping_columns(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--companies", type=Path, required=True)
    parser.add_argument("--adcodes", type=Path)
    parser.add_argument("--code-col")
    parser.add_argument("--name-col")
    parser.add_argument("--alias-col")
    parser.add_argument("--address-col")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    map_parser = subparsers.add_parser("map-companies", help="create company--city crosswalk")
    add_mapping_columns(map_parser)
    map_parser.add_argument("--output", type=Path, required=True)
    map_parser.add_argument("--audit-output", type=Path, required=True)
    map_parser.set_defaults(func=command_map)
    net_parser = subparsers.add_parser("build-network", help="aggregate company links by city/year")
    add_mapping_columns(net_parser)
    net_parser.add_argument("--links", type=Path, required=True)
    net_parser.add_argument("--output-dir", type=Path, required=True)
    net_parser.add_argument("--source-code-col")
    net_parser.add_argument("--target-code-col")
    net_parser.add_argument("--source-name-col")
    net_parser.add_argument("--target-name-col")
    net_parser.add_argument("--year-col")
    net_parser.add_argument("--weight-col")
    net_parser.add_argument("--patent-col")
    net_parser.add_argument("--directed", action="store_true", help="retain citation direction")
    net_parser.add_argument("--include-intra-city", action="store_true")
    net_parser.set_defaults(func=command_network)
    coapp = subparsers.add_parser(
        "build-coapplication", help="build an undirected city patent co-application network"
    )
    add_mapping_columns(coapp)
    coapp.add_argument("--links", type=Path, required=True)
    coapp.add_argument("--repaco", type=Path)
    coapp.add_argument("--output-dir", type=Path, required=True)
    coapp.add_argument("--source-code-col")
    coapp.add_argument("--target-code-col")
    coapp.add_argument("--source-name-col")
    coapp.add_argument("--target-name-col")
    coapp.add_argument("--target-address-col")
    coapp.add_argument("--year-col")
    coapp.add_argument("--date-col")
    coapp.add_argument("--patent-col")
    coapp.add_argument("--include-intra-city", action="store_true")
    coapp.set_defaults(func=command_coapplication)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
