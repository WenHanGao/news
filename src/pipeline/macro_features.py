import os
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Callable, Optional, Literal

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class Data(BaseModel):
    file_path: str
    var_name: str
    load_var_type: Literal["float", "str"]
    preprocess_fn: Optional[Callable] = lambda df: df


def preprocess_consumer_sentiment_index_michigan(df: pd.DataFrame) -> pd.DataFrame:
    df["UMCSENT"] = df["UMCSENT"].apply(lambda x: 0.0 if x == "." else float(x))
    return df


FEATURE_DICTIONARY: Dict[str, List[Data]] = {
    "consumer_confidence": [
        Data(
            file_path="data/macro_data/consumer_opinion_confidence_indicator_OECD.csv",
            var_name="CSCICP03USM665S",
            load_var_type="float",
        )
    ],
    "consumer_sentiment": [
        Data(
            file_path="data/macro_data/consumer_sentiment_index_michigan.csv",
            var_name="UMCSENT",
            load_var_type="str",
            preprocess_fn=preprocess_consumer_sentiment_index_michigan,
        )
    ],
    "equity_volatility": [
        Data(
            file_path="data/macro_data/Equity_market_volatility_tracker.csv",
            var_name="EMVFINCRISES",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Equity_market_volatility_tracker_financial_regulation.csv",
            var_name="EMVFINREG",
            load_var_type="float",
        ),
    ],
    "financial_uncertainty": [
        Data(
            file_path="data/macro_data/lmn_1_month_financial_uncertainty.csv",
            var_name="LMNUF1M",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/lmn_3_month_financial_uncertainty.csv",
            var_name="LMNUF3M",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/lmn_12_month_financial_uncertainty.csv",
            var_name="LMNUF12M",
            load_var_type="float",
        ),
    ],
    "economic_policy_uncertainty": [
        Data(
            file_path="data/macro_data/USEPUINDXD_Economic_pol_uncertainty.csv",
            var_name="USEPUINDXD",
            load_var_type="float",
        )
    ],
    "us_corp_bond_yield_spread": [
        Data(
            file_path="data/macro_data/US Corporate Bond Yield Spread.csv",
            var_name="LUACTRUU Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Corporate Bond Yield Spread(1-3 year).csv",
            var_name="LF99TRUU Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Corporate Bond Yield Spread(3-5 year).csv",
            var_name="BUS3TRUU Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Corporate Bond Yield Spread(5-7 year).csv",
            var_name="I13282US Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Corporate Bond Yield Spread(7-10 year).csv",
            var_name="I13283US Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Corporate Bond Yield Spread(10+ year).csv",
            var_name="I13284US Index",
            load_var_type="float",
        ),
    ],
    "us_generic_govt_yield": [
        Data(
            file_path="data/macro_data/US Generic Govt 3 Month Yield.csv",
            var_name="USGG3M Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Generic Govt 6 Month Yield.csv",
            var_name="USGG6M Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Generic Govt 12 Month Yield.csv",
            var_name="USGG12M Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Generic Govt 2 Year Yield.csv",
            var_name="USGG2YR Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Generic Govt 3 Year Yield.csv",
            var_name="USGG3YR Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Generic Govt 5 Year Yield.csv",
            var_name="USGG5YR Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Generic Govt 7 Year Yield.csv",
            var_name="USGG7YR Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/US Generic Govt 10 Year Yield.csv",
            var_name="USGG10YR Index",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/10year_treasury_yield.csv",
            var_name="10-Year Treasury bond yield",
            load_var_type="float",
        ),
    ],
    "market_stock": [
        Data(
            file_path="data/macro_data/S&P 500 Index.csv",
            var_name="SPX",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Australia_ALL ORDINARIES INDX AS30 Index.csv",
            var_name="AS30",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/China_SHANGHAI SE COMPOSITE IX.csv",
            var_name="SHCOMP",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Hong Kon_HANG SENG INDEX.csv",
            var_name="HIS",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/India_BSE SENSEX 30 INDEX.csv",
            var_name="SENSEX",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Indonesia_JAKARTA COMPOSITE INDEX.csv",
            var_name="JCI",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Japan_NIKKEI 500.csv",
            var_name="NKY500",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Malaysia_FTSE Bursa Malaysia KLCI.csv",
            var_name="FBMKLCI",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Philippines_PSEi - PHILIPPINE SE IDX.csv",
            var_name="PCOMP",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Singapore_STRAITS TIMES INDEX.csv",
            var_name="FSSTI",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Singapore_STRAITS TIMES OLD INDEX.csv",
            var_name="STIOLD",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/South Korea_KOSPI INDEX.csv",
            var_name="KOSPI",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Taiwan_TAIWAN TAIEX INDEX.csv",
            var_name="TWSE",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/Thailand_STOCK EXCH OF THAI INDEX.csv",
            var_name="SET",
            load_var_type="float",
        ),
    ],
    "GDP": [
        Data(file_path="data/macro_data/gdp.csv", var_name="GDP", load_var_type="float")
    ],
    "price_index": [
        Data(
            file_path="data/macro_data/cpi.csv",
            var_name="CPIAUCSL",
            load_var_type="float",
        ),
        Data(
            file_path="data/macro_data/ppi.csv",
            var_name="PPIACO",
            load_var_type="float",
        ),
    ],
    "money": [
        Data(file_path="data/macro_data/m1.csv", var_name="M1", load_var_type="float"),
        Data(file_path="data/macro_data/m2.csv", var_name="M2", load_var_type="float"),
    ],
    "unemployment": [
        Data(
            file_path="data/macro_data/unemployment.csv",
            var_name="Unemployment rate",
            load_var_type="float",
        )
    ],
    "interest_rate": [
        Data(
            file_path="data/macro_data/interest_rate.csv",
            var_name="Interest rate",
            load_var_type="float",
        )
    ],
    "housing": [
        Data(
            file_path="data/macro_data/housing.csv",
            var_name="Housing price index",
            load_var_type="float",
        )
    ],
}


def load_data(data: Data) -> pd.DataFrame:
    var_type = float if data.load_var_type == "float" else str
    df = pd.read_csv(
        os.path.join(BASE_PATH, data.file_path),
        parse_dates=["DATE"],
        dtype={data.var_name: var_type},
    )
    df_preprocessed = data.preprocess_fn(df)
    return df_preprocessed
