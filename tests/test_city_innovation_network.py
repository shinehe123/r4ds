from argparse import Namespace

import pandas as pd

from city_innovation_network import (
    AddressMapper, aggregate_city_network, default_adcodes_path,
    map_companies, prepare_coapplication_edges, prepare_company_edges,
)


def mapper() -> AddressMapper:
    return AddressMapper(default_adcodes_path())


def test_address_mapping_explicit_alias_county_and_municipality():
    companies = pd.DataFrame({
        "code": ["1", "2", "3", "4", "5", "6"],
        "name": ["甲公司", "乙公司", "丙公司", "丁公司", "戊公司", "己公司"],
        "aliases": ["甲", "乙", "丙", "丁", "戊", "己"],
        "address": [
            "广东省深圳市南山区科技园",
            "吉林省长春汽车经济技术开发区",
            "山东省东阿县阿胶街78号",
            "中国(上海)自由贸易试验区张江路665号",
            "福州市马尾区儒江西路6号",
            "江苏省海安市黄海大道268号",
        ],
    })
    result = map_companies(companies, mapper())
    assert result.city.tolist() == [
        "深圳市", "长春市", "聊城市", "上海市", "福州市", "南通市",
    ]
    assert result.company_code.tolist() == [
        "000001", "000002", "000003", "000004", "000005", "000006",
    ]
    assert result.city_code.notna().all()


def test_network_aggregation_and_audit():
    companies = pd.DataFrame({
        "code": ["1", "2", "3"],
        "name": ["甲科技股份有限公司", "乙科技股份有限公司", "丙科技股份有限公司"],
        "aliases": ["甲科技", "乙科技", "丙科技"],
        "address": ["广东省深圳市南山区", "北京市海淀区", "广东省深圳市福田区"],
    })
    crosswalk = map_companies(companies, mapper())
    links = pd.DataFrame({
        "source_code": ["1", "1", "3", "9"],
        "target_code": ["2", "2", "1", "2"],
        "year": [2020, 2020, 2020, 2020],
        "weight": [2, 3, 1, 4],
    })
    args = Namespace(
        source_code_col=None, target_code_col=None, source_name_col=None,
        target_name_col=None, year_col=None, weight_col=None, patent_col=None,
        include_intra_city=False,
    )
    valid, audit = prepare_company_edges(links, crosswalk, args)
    assert len(valid) == 2
    assert audit.exclusion_reason.value_counts().to_dict() == {
        "included": 2, "intra_city": 1, "unmapped_source": 1,
    }
    edges, nodes, evolution = aggregate_city_network(valid, crosswalk, directed=False)
    assert len(edges) == 1
    assert edges.iloc[0].weight == 5
    assert len(nodes) == 2
    assert evolution.iloc[0].total_weight == 5


def test_coapplication_deduplication_address_mapping_and_group_exclusion():
    companies = pd.DataFrame({
        "code": ["1", "2"],
        "name": ["甲科技股份有限公司", "乙科技股份有限公司"],
        "aliases": ["甲科技", "乙科技"],
        "address": ["广东省深圳市南山区", "北京市海淀区"],
    })
    location_mapper = mapper()
    crosswalk = map_companies(companies, location_mapper)
    links = pd.DataFrame({
        "source_code": ["1", "1", "1", "1", "1"],
        "target_code": [None, None, "2", "2", None],
        "target_name": ["苏州研究院", "甲集团成员有限公司", "乙科技", "乙科技", "深圳实验室"],
        "target_address": ["江苏省苏州市", "北京市海淀区", None, None, "广东省深圳市"],
        "patent_id": ["P1", "P2", "P3", "P3", "P4"],
        "year": [2020] * 5,
    })
    repaco = pd.DataFrame({
        "code": ["1"], "name": ["甲集团成员有限公司"], "Relation": ["02"],
    })
    args = Namespace(
        source_code_col=None, target_code_col=None, source_name_col=None,
        target_name_col=None, target_address_col=None, patent_col=None,
        year_col=None, date_col=None, include_intra_city=False,
    )
    valid, audit = prepare_coapplication_edges(
        links, crosswalk, location_mapper, repaco, args
    )
    assert len(valid) == 2
    assert set(valid.patent_id) == {"P1", "P3"}
    assert audit.exclusion_reason.value_counts().to_dict() == {
        "included": 3, "same_corporate_group": 1, "intra_city": 1,
    }
