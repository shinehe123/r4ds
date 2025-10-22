#!/usr/bin/env python3
"""构建基于上市公司专利引用的地级市创新合作网络的脚本。

该脚本使用两份 Excel 数据：
1. ``专利引用数据.xlsx``: 记录上市公司层面的专利引用信息；
2. ``上市公司基本信息库(2022年更新).xlsx``: 记录上市公司基本信息（包括省份、地级市等）。

输出包括：
- 城市层面的引用边表、节点表；
- 各年份网络演化指标；
- 文字分析报告；
- 时间演化折线图以及某一年核心城市网络可视化图。

脚本默认分析 2020 年作为示例年份，可通过修改 ``TARGET_YEAR`` 调整。
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==================== 全局配置 ====================
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CITATION_FILE = "专利引用数据.xlsx"
COMPANY_FILE = "上市公司基本信息库(2022年更新).xlsx"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_YEAR = 2020

# ==================== 工具函数 ====================

def normalize_code(code: object) -> Optional[str]:
    """标准化证券代码，补齐位数并移除无效字符。"""
    if pd.isna(code):
        return None
    code_str = str(code).strip()
    if not code_str:
        return None
    if code_str.lower() in {"nan", "none", "null"}:
        return None
    # 去掉 Excel 导致的 ".0"
    code_str = re.sub(r"\.0+$", "", code_str)
    if code_str.isdigit():
        return code_str.zfill(6)
    return code_str


def clean_company_name(name: str) -> str:
    """移除常见后缀的公司名称，用于模糊匹配。"""
    if not name:
        return ""
    name = str(name).strip()
    patterns = (
        "股份有限公司",
        "有限责任公司",
        "有限公司",
        "集团股份有限公司",
        "集团有限公司",
        "集团股份",
        "股份公司",
        "公司",
    )
    for pattern in patterns:
        if name.endswith(pattern):
            name = name[: -len(pattern)]
            break
    return name.strip()


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """从候选列中返回第一个存在的列名。"""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def format_city(province: Optional[str], city: Optional[str]) -> Optional[str]:
    """组合省份与地级市，生成唯一的地级市标识。"""
    if pd.isna(province):
        province = ""
    if pd.isna(city):
        city = ""
    province = str(province).strip()
    city = str(city).strip()
    if not province and not city:
        return None
    if province and city:
        return f"{province}·{city}"
    return city or province


class CompanyIndex:
    """上市公司信息索引，便于通过公司名称或代码检索信息。"""

    def __init__(self, df: pd.DataFrame) -> None:
        if "股票代码" not in df.columns:
            raise ValueError("上市公司基本信息数据缺少'股票代码'列")

        df = df.copy()
        df["股票代码"] = df["股票代码"].map(normalize_code)
        self.df = df

        name_columns = [
            col
            for col in ["公司中文名称", "证券简称", "证券名称", "公司名称", "简称"]
            if col in df.columns
        ]

        exact_map: Dict[str, set[str]] = defaultdict(set)
        clean_map: Dict[str, set[str]] = defaultdict(set)

        for _, row in df.iterrows():
            code = row["股票代码"]
            if not code:
                continue
            for col in name_columns:
                raw_name = row[col]
                if pd.isna(raw_name):
                    continue
                raw_name_str = str(raw_name).strip()
                if not raw_name_str:
                    continue
                exact_map[raw_name_str].add(code)
                cleaned = clean_company_name(raw_name_str)
                if cleaned:
                    clean_map[cleaned].add(code)

        self.exact_map = {k: list(v) for k, v in exact_map.items()}
        self.clean_map = {k: list(v) for k, v in clean_map.items()}

        self.province_col = first_existing_column(df, ["省份", "所属省份", "省"])
        self.city_col = first_existing_column(
            df,
            [
                "地级市",
                "地级行政区划",
                "城市",
                "地市",
                "地级市(2022年)",
                "地级市名称",
            ],
        )

        location_cols = ["股票代码"]
        if self.province_col:
            location_cols.append(self.province_col)
        if self.city_col:
            location_cols.append(self.city_col)

        self.location_df = df[location_cols].drop_duplicates()
        if self.province_col or self.city_col:
            self.location_df["city_id"] = self.location_df.apply(
                lambda row: format_city(
                    row.get(self.province_col, "") if self.province_col else "",
                    row.get(self.city_col, "") if self.city_col else "",
                ),
                axis=1,
            )
        else:
            self.location_df["city_id"] = np.nan

    def get_code_by_name(self, name: object) -> Optional[str]:
        if pd.isna(name):
            return None
        name_str = str(name).strip()
        if not name_str:
            return None

        # 精确匹配
        if name_str in self.exact_map:
            return self.exact_map[name_str][0]

        cleaned = clean_company_name(name_str)
        if cleaned in self.exact_map:
            return self.exact_map[cleaned][0]
        if cleaned in self.clean_map:
            return self.clean_map[cleaned][0]

        # 进一步模糊匹配
        candidates = [
            code
            for key, codes in self.clean_map.items()
            if cleaned and len(cleaned) > 2 and (cleaned in key or key in cleaned)
            for code in codes
        ]
        return candidates[0] if candidates else None

    def get_company_info(self, code: str) -> Optional[pd.Series]:
        if not code:
            return None
        matched = self.df[self.df["股票代码"] == code]
        if matched.empty:
            return None
        return matched.iloc[0]

    def get_location(self, code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        info = self.get_company_info(code)
        if info is None:
            return None, None, None
        province = info.get(self.province_col) if self.province_col else None
        city = info.get(self.city_col) if self.city_col else None
        city_id = format_city(province, city)
        return province, city, city_id

    def build_city_company_counts(self) -> pd.Series:
        if "city_id" not in self.location_df.columns:
            return pd.Series(dtype=int)
        return (
            self.location_df.dropna(subset=["city_id"])
            .groupby("city_id")["股票代码"]
            .nunique()
            .sort_values(ascending=False)
        )


# ==================== 数据加载与清洗 ====================

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 80)
    print("地级市创新合作网络构建系统".center(70))
    print("=" * 80)

    print("\n📂 Step 1: 加载数据...")
    citation_df = pd.read_excel(CITATION_FILE)
    company_df = pd.read_excel(COMPANY_FILE)
    print(f"   ✅ 专利引用数据加载成功: {len(citation_df)} 条记录")
    print(f"   ✅ 上市公司数据加载成功: {len(company_df)} 家公司")

    print("\n   数据预览 - 专利引用数据:")
    preview_cols = citation_df.columns[: min(4, len(citation_df.columns))]
    print(citation_df[preview_cols].head(3).to_string(index=False))
    return citation_df, company_df


def clean_citation_data(citation_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("🧹 Step 2: 数据清洗与标准化...")

    column_mapping = {
        "证券代码": "证券代码",
        "被引用年度": "被引用年度",
        "引用方": "引用方",
        "被引用次数": "被引用次数",
    }
    citation_df = citation_df.rename(columns=column_mapping)

    required_cols = list(column_mapping.values())
    missing_cols = [col for col in required_cols if col not in citation_df.columns]
    if missing_cols:
        raise ValueError(f"专利引用数据缺少必要列: {missing_cols}")

    citation_df = citation_df.dropna(subset=["证券代码", "引用方"])
    citation_df["证券代码"] = citation_df["证券代码"].map(normalize_code)
    citation_df["引用方"] = citation_df["引用方"].astype(str).str.strip()
    citation_df["被引用年度"] = pd.to_numeric(
        citation_df["被引用年度"], errors="coerce"
    ).astype("Int64")
    citation_df["被引用次数"] = (
        pd.to_numeric(citation_df["被引用次数"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    exclude_keywords = {
        "上市公司本身小计",
        "子公司小计",
        "合营联营公司小计",
        "上市公司及子公司合营联营公司合计",
    }
    citation_df = citation_df[~citation_df["引用方"].isin(exclude_keywords)].copy()

    print(f"   ✅ 清洗后数据: {len(citation_df)} 条有效记录")
    return citation_df


def match_citing_company_codes(
    citation_df: pd.DataFrame, company_index: CompanyIndex
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("🔗 Step 3: 匹配引用方公司代码...")

    citation_df = citation_df.copy()
    citation_df["引用方代码"] = citation_df["引用方"].apply(company_index.get_code_by_name)

    total_records = len(citation_df)
    matched_records = citation_df["引用方代码"].notna().sum()
    match_rate = matched_records / total_records * 100 if total_records else 0

    print("   ✅ 匹配完成:")
    print(f"      - 总记录数: {total_records}")
    print(f"      - 成功匹配: {matched_records}")
    print(f"      - 匹配率: {match_rate:.2f}%")

    if matched_records < total_records:
        unmatched = (
            citation_df[citation_df["引用方代码"].isna()]["引用方"].value_counts().head(10)
        )
        if not unmatched.empty:
            print("\n   ⚠️  未匹配的引用方示例 (Top 10):")
            for company, count in unmatched.items():
                print(f"      {company}: {int(count)} 次")

    return citation_df


# ==================== 城市层面网络构建 ====================

def build_city_level_edges(
    citation_df: pd.DataFrame, company_index: CompanyIndex
) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print("🌐 Step 4: 构建地级市创新合作网络...")

    df = citation_df.dropna(subset=["引用方代码"]).copy()
    df = df[df["引用方代码"].astype(str).str.len() > 0]

    edges = []
    missing_location = 0

    for _, row in df.iterrows():
        target_code = row["证券代码"]
        source_code = row["引用方代码"]
        if not target_code or not source_code:
            continue

        target_province, target_city, target_city_id = company_index.get_location(
            target_code
        )
        source_province, source_city, source_city_id = company_index.get_location(
            source_code
        )

        if not target_city_id or not source_city_id:
            missing_location += 1
            continue

        edges.append(
            {
                "source_city": source_city_id,
                "target_city": target_city_id,
                "source_province": source_province,
                "source_city_name": source_city,
                "target_province": target_province,
                "target_city_name": target_city,
                "year": int(row["被引用年度"]) if pd.notna(row["被引用年度"]) else None,
                "citation_count": int(row["被引用次数"]),
            }
        )

    if not edges:
        raise ValueError("未能构建任何地级市层面的引用关系，请检查数据与匹配结果。")

    edges_df = pd.DataFrame(edges)
    edges_df = edges_df.dropna(subset=["year"])

    city_network = (
        edges_df.groupby(["source_city", "target_city", "year"], as_index=False)
        .agg(
            citation_count=("citation_count", "sum"),
            source_province=("source_province", "first"),
            source_city_name=("source_city_name", "first"),
            target_province=("target_province", "first"),
            target_city_name=("target_city_name", "first"),
        )
    )

    # 排除同城自循环，聚焦跨城市合作
    city_network = city_network[city_network["source_city"] != city_network["target_city"]]

    total_citations = int(city_network["citation_count"].sum()) if not city_network.empty else 0
    nodes = set(city_network["source_city"]) | set(city_network["target_city"])
    avg_weight = city_network["citation_count"].mean() if not city_network.empty else 0

    print("   ✅ 网络构建完成:")
    print(f"      - 地级市节点数: {len(nodes)}")
    print(f"      - 城市间边数: {len(city_network)}")
    print(f"      - 总引用次数: {total_citations}")
    print(f"      - 平均每条边引用次数: {avg_weight:.2f}")
    print(f"      - 缺失地理信息记录数: {missing_location}")

    if city_network.empty:
        print("      ⚠️ 城市间合作数据为空（可能仅存在同城引用或缺少地理信息）")

    return city_network


def analyze_city_network(
    edges_df: pd.DataFrame, year: int
) -> Tuple[Optional[Dict[str, object]], list, list, Optional[nx.DiGraph]]:
    df_year = edges_df[edges_df["year"] == year]
    if df_year.empty:
        print(f"   ⚠️  {year} 年无城市层面的引用数据")
        return None, [], [], None

    G = nx.DiGraph()
    for _, row in df_year.iterrows():
        G.add_edge(
            row["source_city"],
            row["target_city"],
            weight=row["citation_count"],
        )

    if G.number_of_nodes() == 0:
        print(f"   ⚠️  {year} 年网络为空")
        return None, [], [], G

    avg_degree = (
        sum(dict(G.degree()).values()) / G.number_of_nodes()
        if G.number_of_nodes() > 0
        else 0
    )

    stats = {
        "年份": year,
        "节点数": G.number_of_nodes(),
        "边数": G.number_of_edges(),
        "平均度": round(avg_degree, 2),
        "网络密度": round(nx.density(G), 4),
        "弱连通分量数": nx.number_weakly_connected_components(G),
        "强连通分量数": nx.number_strongly_connected_components(G),
    }

    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))
    top_in = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
    top_out = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]

    return stats, top_in, top_out, G


def compute_time_evolution(edges_df: pd.DataFrame) -> pd.DataFrame:
    if edges_df.empty:
        print("\n" + "=" * 80)
        print("⏰ Step 7: 时间演化分析...")
        print("   ⚠️ 城市间引用数据为空，无法进行时间演化分析")
        return pd.DataFrame(
            columns=["year", "nodes", "edges", "avg_degree", "density", "total_citations"]
        )

    years = sorted(edges_df["year"].dropna().unique())
    if not years:
        print("\n" + "=" * 80)
        print("⏰ Step 7: 时间演化分析...")
        print("   ⚠️ 缺少年份信息，无法进行时间演化分析")
        return pd.DataFrame(
            columns=["year", "nodes", "edges", "avg_degree", "density", "total_citations"]
        )
    print("\n" + "=" * 80)
    print("⏰ Step 7: 时间演化分析...")
    print(f"   📅 数据覆盖年份: {years[0]} - {years[-1]}")

    records = []
    for year in years:
        df_year = edges_df[edges_df["year"] == year]
        G_year = nx.DiGraph()
        for _, row in df_year.iterrows():
            G_year.add_edge(
                row["source_city"],
                row["target_city"],
                weight=row["citation_count"],
            )

        avg_degree = (
            sum(dict(G_year.degree()).values()) / G_year.number_of_nodes()
            if G_year.number_of_nodes() > 0
            else 0
        )

        records.append(
            {
                "year": year,
                "nodes": G_year.number_of_nodes(),
                "edges": G_year.number_of_edges(),
                "avg_degree": avg_degree,
                "density": nx.density(G_year) if G_year.number_of_nodes() > 0 else 0,
                "total_citations": int(df_year["citation_count"].sum()),
            }
        )

    evolution_df = pd.DataFrame(records)
    print("\n   📊 网络演化趋势:")
    print(evolution_df.to_string(index=False))
    return evolution_df


def plot_time_evolution(evolution_df: pd.DataFrame) -> None:
    if evolution_df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("地级市创新合作网络时间演化分析", fontsize=16, fontweight="bold")

    axes[0, 0].plot(
        evolution_df["year"],
        evolution_df["nodes"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="#2E86AB",
    )
    axes[0, 0].set_title("节点数（地级市数量）变化", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("年份")
    axes[0, 0].set_ylabel("节点数")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        evolution_df["year"],
        evolution_df["edges"],
        marker="s",
        linewidth=2,
        markersize=6,
        color="#A23B72",
    )
    axes[0, 1].set_title("边数（城市合作关系）变化", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("年份")
    axes[0, 1].set_ylabel("边数")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(
        evolution_df["year"],
        evolution_df["avg_degree"],
        marker="^",
        linewidth=2,
        markersize=6,
        color="#F18F01",
    )
    axes[0, 2].set_title("平均度变化", fontsize=12, fontweight="bold")
    axes[0, 2].set_xlabel("年份")
    axes[0, 2].set_ylabel("平均度")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(
        evolution_df["year"],
        evolution_df["density"],
        marker="d",
        linewidth=2,
        markersize=6,
        color="#C73E1D",
    )
    axes[1, 0].set_title("网络密度变化", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("年份")
    axes[1, 0].set_ylabel("密度")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        evolution_df["year"],
        evolution_df["total_citations"],
        marker="*",
        linewidth=2,
        markersize=8,
        color="#6A994E",
    )
    axes[1, 1].set_title("总引用次数变化", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("年份")
    axes[1, 1].set_ylabel("引用次数")
    axes[1, 1].grid(True, alpha=0.3)

    growth_rate = (evolution_df["edges"].pct_change() * 100).iloc[1:]
    axes[1, 2].bar(
        evolution_df["year"].iloc[1:],
        growth_rate.fillna(0),
        color="#BC4B51",
        alpha=0.7,
    )
    axes[1, 2].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    axes[1, 2].set_title("边数增长率 (%)", fontsize=12, fontweight="bold")
    axes[1, 2].set_xlabel("年份")
    axes[1, 2].set_ylabel("增长率 (%)")
    axes[1, 2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "地级市创新合作网络_时间演化.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n   ✅ 演化趋势图已保存: {output_path}")
    plt.close(fig)


def visualize_city_network(G: nx.DiGraph, year: int) -> None:
    if G is None or G.number_of_nodes() == 0:
        return

    degree_centrality = nx.degree_centrality(G)
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:40]
    sub_nodes = [node for node, _ in top_nodes]
    G_sub = G.subgraph(sub_nodes).copy()

    print(f"   🔄 绘制 {year} 年核心城市创新合作网络（Top 40 节点）...")

    pos = nx.spring_layout(G_sub, k=3, iterations=50, seed=42)
    in_degrees = dict(G_sub.in_degree(weight="weight"))
    node_sizes = [max(in_degrees.get(node, 1) * 10, 100) for node in G_sub.nodes()]
    node_colors = [degree_centrality.get(node, 0) for node in G_sub.nodes()]

    plt.figure(figsize=(24, 20))
    nx.draw_networkx_edges(
        G_sub,
        pos,
        edge_color="gray",
        alpha=0.2,
        arrows=True,
        arrowsize=15,
        width=0.8,
        arrowstyle="->",
    )
    nodes = nx.draw_networkx_nodes(
        G_sub,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        alpha=0.9,
        edgecolors="black",
        linewidths=1.2,
    )

    labels = {node: node for node in G_sub.nodes()}
    nx.draw_networkx_labels(
        G_sub,
        pos,
        labels,
        font_size=9,
        font_color="black",
        font_weight="bold",
    )

    plt.colorbar(nodes, label="度中心性", shrink=0.8)
    plt.title(
        f"地级市创新合作网络（{year} 年 Top 40 核心城市）",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    plt.axis("off")
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"地级市创新合作网络可视化_{year}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"   ✅ 网络可视化图已保存: {output_path}")
    plt.close()


# ==================== 报告与结果保存 ====================

def build_node_table(
    edges_df: pd.DataFrame,
    company_index: CompanyIndex,
    overall_graph: nx.DiGraph,
) -> pd.DataFrame:
    company_counts = company_index.build_city_company_counts()

    in_citations = (
        edges_df.groupby("target_city")["citation_count"].sum().rename("in_citations")
    )
    out_citations = (
        edges_df.groupby("source_city")["citation_count"].sum().rename("out_citations")
    )

    unique_inbound = (
        edges_df.groupby("target_city")["source_city"].nunique().rename("unique_inbound_cities")
    )
    unique_outbound = (
        edges_df.groupby("source_city")["target_city"].nunique().rename(
            "unique_outbound_cities"
        )
    )

    location_lookup: Dict[str, Dict[str, Optional[str]]] = {}
    if "city_id" in company_index.location_df.columns:
        province_col = company_index.province_col
        city_col = company_index.city_col
        for _, row in company_index.location_df.dropna(subset=["city_id"]).iterrows():
            city_id = row["city_id"]
            if city_id in location_lookup:
                continue
            province = row[province_col] if province_col and province_col in row.index else None
            city_name = row[city_col] if city_col and city_col in row.index else None
            location_lookup[city_id] = {
                "province": province,
                "city_name": city_name,
            }

    records = []
    for city in sorted(set(edges_df["source_city"]) | set(edges_df["target_city"])):
        info = location_lookup.get(city, {})
        province = info.get("province")
        city_name = info.get("city_name") or city

        company_count = int(company_counts.get(city, 0)) if not company_counts.empty else 0
        records.append(
            {
                "city_id": city,
                "province": province,
                "city_name": city_name,
                "listed_company_count": company_count,
                "in_citations": int(in_citations.get(city, 0)),
                "out_citations": int(out_citations.get(city, 0)),
                "total_citations": int(in_citations.get(city, 0) + out_citations.get(city, 0)),
                "unique_inbound_cities": int(unique_inbound.get(city, 0)),
                "unique_outbound_cities": int(unique_outbound.get(city, 0)),
            }
        )

    nodes_df = pd.DataFrame(records)

    if overall_graph.number_of_nodes() > 0:
        betweenness = nx.betweenness_centrality(overall_graph, weight="weight")
        degree_centrality = nx.degree_centrality(overall_graph)
        nodes_df["betweenness_centrality"] = nodes_df["city_id"].map(betweenness).fillna(0)
        nodes_df["degree_centrality"] = nodes_df["city_id"].map(degree_centrality).fillna(0)
    else:
        nodes_df["betweenness_centrality"] = 0
        nodes_df["degree_centrality"] = 0

    nodes_df = nodes_df.sort_values("total_citations", ascending=False)
    return nodes_df


def save_outputs(
    city_network: pd.DataFrame,
    nodes_table: pd.DataFrame,
    evolution_df: pd.DataFrame,
    stats: Optional[Dict[str, object]],
    top_in: list,
    top_out: list,
    top_bridges: list,
) -> None:
    edges_output = OUTPUT_DIR / "地级市创新合作网络_边表.csv"
    city_network.to_csv(edges_output, index=False, encoding="utf-8-sig")
    print(f"   ✅ 城市层面边表已保存: {edges_output}")

    nodes_output = OUTPUT_DIR / "地级市创新合作网络_节点表.csv"
    nodes_table.to_csv(nodes_output, index=False, encoding="utf-8-sig")
    print(f"   ✅ 城市层面节点表已保存: {nodes_output}")

    evolution_output = OUTPUT_DIR / "地级市创新合作网络_时间演化.csv"
    evolution_df.to_csv(evolution_output, index=False, encoding="utf-8-sig")
    print(f"   ✅ 时间演化数据已保存: {evolution_output}")

    report_output = OUTPUT_DIR / "地级市创新合作网络_分析报告.txt"
    with report_output.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("地级市创新合作网络分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("一、网络基本统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"地级市节点数: {len(set(city_network['source_city']) | set(city_network['target_city']))}\n")
        f.write(f"城市间边数: {len(city_network)}\n")
        f.write(f"总引用次数: {int(city_network['citation_count'].sum())}\n\n")

        if stats:
            f.write("二、{0} 年网络指标\n".format(stats.get("年份")))
            f.write("-" * 80 + "\n")
            for key, value in stats.items():
                if key == "年份":
                    continue
                f.write(f"{key}: {value}\n")
            f.write("\n")

        f.write("三、被引用最频繁的地级市 (Top 10)\n")
        f.write("-" * 80 + "\n")
        for i, (city, count) in enumerate(top_in, 1):
            f.write(f"{i}. {city}: {int(count)} 次\n")
        f.write("\n")

        f.write("四、引用最活跃的地级市 (Top 10)\n")
        f.write("-" * 80 + "\n")
        for i, (city, count) in enumerate(top_out, 1):
            f.write(f"{i}. {city}: {int(count)} 次\n")
        f.write("\n")

        f.write("五、技术知识流动关键枢纽 (Top 10)\n")
        f.write("-" * 80 + "\n")
        for i, (city, score) in enumerate(top_bridges, 1):
            f.write(f"{i}. {city}: {score:.4f}\n")

    print(f"   ✅ 分析报告已保存: {report_output}")


# ==================== 主流程 ====================

def main() -> None:
    citation_df, company_df = load_data()
    citation_df = clean_citation_data(citation_df)
    company_index = CompanyIndex(company_df)
    citation_df = match_citing_company_codes(citation_df, company_index)
    city_network = build_city_level_edges(citation_df, company_index)

    print("\n" + "=" * 80)
    print("📊 Step 5: 地级市网络深度分析...")

    stats, top_in, top_out, G_year = analyze_city_network(city_network, TARGET_YEAR)
    if stats:
        print(f"\n   📈 {stats['年份']} 年网络统计指标:")
        for key, value in stats.items():
            if key != "年份":
                print(f"      - {key}: {value}")

    if top_in:
        print(f"\n   🏆 被引用最频繁的地级市 (Top 10):")
        for i, (city, count) in enumerate(top_in, 1):
            print(f"      {i}. {city}: {int(count)} 次")

    if top_out:
        print(f"\n   🔥 引用最活跃的地级市 (Top 10):")
        for i, (city, count) in enumerate(top_out, 1):
            print(f"      {i}. {city}: {int(count)} 次")

    top_bridges: list[Tuple[str, float]] = []
    if G_year and G_year.number_of_nodes() > 0:
        print("\n" + "=" * 80)
        print("🏘️  Step 6: 识别城市创新社区...")
        betweenness = nx.betweenness_centrality(G_year, weight="weight")
        top_bridges = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]

        if top_bridges:
            print("\n   🌉 技术知识流动的枢纽城市 (Top 10):")
            for i, (city, score) in enumerate(top_bridges, 1):
                print(f"      {i}. {city}: {score:.4f}")

        G_undirected = G_year.to_undirected()
        communities = list(nx.community.greedy_modularity_communities(G_undirected))
        print(f"\n   ✅ 检测到 {len(communities)} 个城市创新社区")
        for i, community in enumerate(sorted(communities, key=len, reverse=True)[:5], 1):
            members = sorted(list(community))
            print(f"\n      社区 {i} (规模: {len(members)} 座城市):")
            for city in members[:5]:
                print(f"         - {city}")
            if len(members) > 5:
                print(f"         ... 还有 {len(members) - 5} 座城市")

    evolution_df = compute_time_evolution(city_network)
    plot_time_evolution(evolution_df)

    if G_year and G_year.number_of_nodes() > 0:
        print("\n" + "=" * 80)
        print("🎨 Step 8: 网络可视化...")
        visualize_city_network(G_year, TARGET_YEAR)

    overall_edges = (
        city_network.groupby(["source_city", "target_city"], as_index=False)
        .agg({"citation_count": "sum"})
    )
    G_overall = nx.DiGraph()
    for _, row in overall_edges.iterrows():
        G_overall.add_edge(
            row["source_city"],
            row["target_city"],
            weight=row["citation_count"],
        )

    nodes_table = build_node_table(city_network, company_index, G_overall)

    print("\n" + "=" * 80)
    print("💾 Step 10: 保存分析结果...")
    save_outputs(city_network, nodes_table, evolution_df, stats, top_in, top_out, top_bridges)

    print("\n" + "=" * 80)
    print("✅ 地级市创新合作网络构建与分析完成！".center(70))
    print("=" * 80)

    print("\n📁 生成的文件 (位于 outputs/):")
    print("   1. 地级市创新合作网络_边表.csv")
    print("   2. 地级市创新合作网络_节点表.csv")
    print("   3. 地级市创新合作网络_时间演化.csv")
    print("   4. 地级市创新合作网络_分析报告.txt")
    print("   5. 地级市创新合作网络_时间演化.png")
    if G_year and G_year.number_of_nodes() > 0:
        print(f"   6. 地级市创新合作网络可视化_{TARGET_YEAR}.png")


if __name__ == "__main__":
    main()
