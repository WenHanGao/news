import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Callable, Optional, Literal
import cvxopt

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

train_features = pd.read_excel(
    os.path.join(BASE_PATH, 'data/recovery_rate/train_features2.xlsx'),
    dtype=float
)
train_labels = pd.read_excel(
    os.path.join(BASE_PATH, 'data/recovery_rate/train_labels2.xlsx')
)


# The instrument characteristics included in the model are 
# (1) the bond’s seniority in the capital structure, 
# (2) the amount of the bond’s trading volume, 
# (3) a dummy variable indicating bonds that have been assigned 
# a rating that is the equivalent of a default. 
# 
# For industry characteristics, we include 
# (1) a dummy variable for the utility industry (this is an issuer characteristics) 
# (2) industry distress dummy variable is based on 
# whether the performance of the industry index was worse than −30% 
# in the year preceding the default. 
# (3) industry distress dummy variable is based on 
# whether the sales growth in the respective industry in the year 
# preceding the default was negative. 
# 
# The macroeconomic variables include 
# (1) the number of defaulted bonds in the respective year, 
# (2) the value of a high-yield index, 
# (3) the change in gross domestic product (GDP), 
# (4) the unemployment rate, and 
# (5) the federal funds rate. 
# The macroeconomic variables are observed in the year preceding the default.

INSTRUMENT_FEATURES = [
    'seniorioty_adj_Junior Unsecured or Junior Subordinated Unsecured',
    'seniorioty_adj_Secured',
    'seniorioty_adj_Senior Secured',
    'seniorioty_adj_Senior Subordinated Unsecured',
    'seniorioty_adj_Senior Unsecured',
    'seniorioty_adj_Subordinated Unsecured',
    'seniorioty_adj_Unsecured',
] # missing tading volume and previous rating history

ISSUER_FEATURES = [
    'Industry_sector_Utilities',
    'Industry_group_Utilities',
]

INDUSTRY_FEATURES = [
]   # missing industry index performance and sale

MACRO_FEATURES = [
    '1-year_GDP_growth',
    'employment_rate',
    'interest_rate',
] # missing number of defaulted bonds in the respective year and high-yield index


ADDITIONAL_INSTRUMENT_FEATURES = [
    'coupon rate',
    'duration_in_years',
    'currency_CAD',
    'currency_CHF',
    'currency_CNY',
    'currency_EUR',
    'currency_GBP',
    'currency_HKD',
    'currency_INR',
    'currency_ISK',
    'currency_JPY',
    'currency_MYR',
    'currency_NOK',
    'currency_SEK',
    'currency_SGD',
    'currency_THB',
    'currency_TWD',
    'currency_USD',
]


ADDITIONAL_INDUSTRY_FEATURES = [
    'sector_domicile_dtd',
    'sector_exchange_dtd',
    'sector_dtd',
    'domicile_subsec_dtd',
    'exch_subsector_dtd',
    'subsector_dtd',
    'PD_1_domicile_sector',
    'PD_3_domicile_sector',
    'PD_12_domicile_sector',
    'PD_1_domicile_subsec',
    'PD_3_domicile_subsec',
    'PD_12_domicile_subsec',
    'PD_1_exch_sector',
    'PD_3_exch_sector',
    'PD_12_exch_sector',
    'PD_1_exch_subsector',
    'PD_3_exch_subsector',
    'PD_12_exch_subsector',
    'PD_1_global_sector',
    'PD_3_global_sector',
    'PD_12_global_sector',
    'PD_1_global_subsector',
    'PD_3_global_subsector',
    'PD_12_global_subsector',
]

ADDITIONAL_MACRO_FEATURES = [
    'SP500 MD',
    'Average daily 1-year SP500 return',
    'Ratio to MA',
    'US Corporate Bond Yield Spread',
    'US Corporate Bond Yield Spread(3-5 year)',
    'US Corporate Bond Yield Spread(5-7 year)',
    'US Corporate Bond Yield Spread(7-10 year)',
    'US Corporate Bond Yield Spread(10+ year)',
    'US Generic Govt 3 Month Yield',
    'US Generic Govt 6 Month Yield',
    'US Generic Govt 12 Month Yield',
    'US Generic Govt 2 Year Yield',
    'US Generic Govt 3 Year Yield',
    'US Generic Govt 5 Year Yield',
    'US Generic Govt 7 Year Yield',
    'US Generic Govt 10 Year Yield',
    'Three_Month_Rate_After_Demean',
    '1-year_CPI_growth',
    '1-year_PPI_growth',
    '1-year_M1_growth',
    '1-year_M2_growth',
    'treasury_rate',
    'economic_policy_uncertainty',
    '1-year_housing',
    'mktcap_median',
    'MA_Index_ratio',
    'PD_1_sector_median',
    'PD_3_sector_median',
    'PD_12_sector_median',
    'dtd_sector_median',
    'dtd_median_exch',
    'PD_1_median_exch',
    'PD_3_median_exch',
    'PD_12_median_exch',
    'PD_1_median_domicile',
    'PD_3_median_domicile',
    'PD_12_median_domicile',
    'equity_market_volatility_tracker_financial_crises',
    'equity_market_volatility_tracker_financial_regulation',
    '1_month_financial_uncertainty',
    '3_month_financial_certainty',
    '12_month_financial_certainty',
    'consumer_sentiment_index',
    'consumer_opinion_confidence_indicator',
]

ADDITIONAL_ISSUER_FEATURES = [
    'marketcap',
    'Stock_Index_Return',
    'DTD_Level',
    'M_Over_B',
    'Sigma',
    'ROA',
    'shares_out_outstanding_shares',
    'relative_size',
    'current ratio',
    'quick_ratio',
    'debt_to_equity',
    'liquidity_ratio',
    'operating_margin',
    'defaulted_in_last_5_years',
    'ROE',
    'Long-term_ratio',
    'last price_new',
    'dtd_sigma',
    'PD_1_pd',
    'PD_2_pd',
    'PD_3_pd',
    'PD_4_pd',
    'PD_5_pd',
    'PD_6_pd',
    'PD_7_pd',
    'PD_8_pd',
    'PD_9_pd',
    'PD_10_pd',
    'PD_11_pd',
    'PD_12_pd',
    'PD_13_pd',
    'PD_14_pd',
    'PD_15_pd',
    'PD_16_pd',
    'PD_17_pd',
    'PD_18_pd',
    'PD_19_pd',
    'PD_20_pd',
    'PD_21_pd',
    'PD_22_pd',
    'PD_23_pd',
    'PD_24_pd',
    'PD_25_pd',
    'PD_26_pd',
    'PD_27_pd',
    'PD_28_pd',
    'PD_29_pd',
    'PD_30_pd',
    'PD_31_pd',
    'PD_32_pd',
    'PD_33_pd',
    'PD_34_pd',
    'PD_35_pd',
    'PD_36_pd',
    'PD_37_pd',
    'PD_38_pd',
    'PD_39_pd',
    'PD_40_pd',
    'PD_41_pd',
    'PD_42_pd',
    'PD_43_pd',
    'PD_44_pd',
    'PD_45_pd',
    'PD_46_pd',
    'PD_47_pd',
    'PD_48_pd',
    'PD_49_pd',
    'PD_50_pd',
    'PD_51_pd',
    'PD_52_pd',
    'PD_53_pd',
    'PD_54_pd',
    'PD_55_pd',
    'PD_56_pd',
    'PD_57_pd',
    'PD_58_pd',
    'PD_59_pd',
    'PD_60_pd',
    'DTD',
    'NI_Over_TA',
    'Size',
    'domicile_country_Argentina',
    'domicile_country_Australia',
    'domicile_country_Bahamas',
    'domicile_country_Belgium',
    'domicile_country_Bermuda',
    'domicile_country_Canada',
    'domicile_country_Cayman Islands',
    'domicile_country_China',
    'domicile_country_Czech Republic',
    'domicile_country_Greece',
    'domicile_country_Hong Kong',
    'domicile_country_Iceland',
    'domicile_country_India',
    'domicile_country_Indonesia',
    'domicile_country_Japan',
    'domicile_country_Luxembourg',
    'domicile_country_Malaysia',
    'domicile_country_Mongolia',
    'domicile_country_Philippines',
    'domicile_country_Poland',
    'domicile_country_Singapore',
    'domicile_country_South Africa',
    'domicile_country_South Korea',
    'domicile_country_Taiwan',
    'domicile_country_Thailand',
    'domicile_country_United Kingdom',
    'domicile_country_United States',
    'exchange_country_Australia',
    'exchange_country_China',
    'exchange_country_Hong Kong',
    'exchange_country_India',
    'exchange_country_Indonesia',
    'exchange_country_Japan',
    'exchange_country_Malaysia',
    'exchange_country_Philippines',
    'exchange_country_Singapore',
    'exchange_country_South Korea',
    'exchange_country_Taiwan',
    'exchange_country_Thailand',
    'exchange_country_United States',
    'Industry_sector_Communications',
    'Industry_sector_Consumer Discretionary',
    'Industry_sector_Consumer Staples',
    'Industry_sector_Energy',
    'Industry_sector_Financials',
    'Industry_sector_Health Care',
    'Industry_sector_Industrials',
    'Industry_sector_Materials',
    'Industry_sector_Real Estate',
    'Industry_sector_Technology',
    'Industry_group_Banking',
    'Industry_group_Consumer Discretionary Products',
    'Industry_group_Consumer Discretionary Services',
    'Industry_group_Consumer Staple Products',
    'Industry_group_Health Care',
    'Industry_group_Industrial Products',
    'Industry_group_Industrial Services',
    'Industry_group_Insurance',
    'Industry_group_Materials',
    'Industry_group_Media',
    'Industry_group_Oil & Gas',
    'Industry_group_Real Estate',
    'Industry_group_Renewable Energy',
    'Industry_group_Retail & Wholesale - Staples',
    'Industry_group_Retail & Whsle - Discretionary',
    'Industry_group_Software & Tech Services',
    'Industry_group_Tech Hardware & Semiconductors',
    'Industry_group_Telecommunications',
    'Industry_subgroup_Advertising & Marketing',
    'Industry_subgroup_Apparel & Textile Products',
    'Industry_subgroup_Automotive',
    'Industry_subgroup_Banking',
    'Industry_subgroup_Beverages',
    'Industry_subgroup_Biotech & Pharma',
    'Industry_subgroup_Cable & Satellite',
    'Industry_subgroup_Chemicals',
    'Industry_subgroup_Commercial Support Services',
    'Industry_subgroup_Construction Materials',
    'Industry_subgroup_Consumer Services',
    'Industry_subgroup_Containers & Packaging',
    'Industry_subgroup_E-Commerce Discretionary',
    'Industry_subgroup_Electric Utilities',
    'Industry_subgroup_Electrical Equipment',
    'Industry_subgroup_Engineering & Construction',
    'Industry_subgroup_Entertainment Content',
    'Industry_subgroup_Food',
    'Industry_subgroup_Forestry, Paper & Wood Products',
    'Industry_subgroup_Gas & Water Utilities',
    'Industry_subgroup_Health Care Facilities & Svcs',
    'Industry_subgroup_Home & Office Products',
    'Industry_subgroup_Home Construction',
    'Industry_subgroup_Household Products',
    'Industry_subgroup_Industrial Intermediate Prod',
    'Industry_subgroup_Industrial Support Services',
    'Industry_subgroup_Insurance',
    'Industry_subgroup_Leisure Facilities & Services',
    'Industry_subgroup_Leisure Products',
    'Industry_subgroup_Machinery',
    'Industry_subgroup_Medical Equipment & Devices',
    'Industry_subgroup_Metals & Mining',
    'Industry_subgroup_Oil & Gas Producers',
    'Industry_subgroup_Oil & Gas Services & Equip',
    'Industry_subgroup_Publishing & Broadcasting',
    'Industry_subgroup_REIT',
    'Industry_subgroup_Real Estate Owners & Developers',
    'Industry_subgroup_Real Estate Services',
    'Industry_subgroup_Renewable Energy',
    'Industry_subgroup_Retail - Consumer Staples',
    'Industry_subgroup_Retail - Discretionary',
    'Industry_subgroup_Semiconductors',
    'Industry_subgroup_Software',
    'Industry_subgroup_Steel',
    'Industry_subgroup_Technology Hardware',
    'Industry_subgroup_Technology Services',
    'Industry_subgroup_Telecommunications',
    'Industry_subgroup_Transportation & Logistics',
    'Industry_subgroup_Transportation Equipment',
    'Industry_subgroup_Wholesale - Consumer Staples',
    'Industry_subgroup_Wholesale - Discretionary',
    'event_type_Bankruptcy Filing',
    'event_type_Default Corp Action',
    'event_type_Delisting',
    'event_type_subcategory_sum_Bankruptcy',
    'event_type_subcategory_sum_Debt Restructuring',
    'event_type_subcategory_sum_Insolvency',
    'event_type_subcategory_sum_Liquidation',
    'event_type_subcategory_sum_Missing Coupon & principal payment',
    'event_type_subcategory_sum_Missing Coupon payment only',
    'event_type_subcategory_sum_Missing Interest payment',
    'event_type_subcategory_sum_Missing Loan payment',
    'event_type_subcategory_sum_Missing Principal payment',
    'event_type_subcategory_sum_Others',
    'event_type_subcategory_sum_Pre-Negotiated Chapter 11',
    'event_type_subcategory_sum_Protection',
    'event_type_subcategory_sum_Receivership',
    'event_type_subcategory_sum_Rehabilitation',
    'event_type_subcategory_sum_Restructuring',
    'defaulted_in_last_6_months',
]

senority_classes = train_features[INSTRUMENT_FEATURES]
z = senority_classes.values
X = train_features[
    ADDITIONAL_INSTRUMENT_FEATURES +
    ISSUER_FEATURES + 
    ADDITIONAL_ISSUER_FEATURES +
    INDUSTRY_FEATURES +
    ADDITIONAL_INDUSTRY_FEATURES +  
    MACRO_FEATURES +
    ADDITIONAL_MACRO_FEATURES
].values
X_normalized = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
y = train_labels['rr1_30'].values

def rbf_kernel(X1, X2, gamma=0.1):
    pairwise_sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * (X1 @ X2.T)
    return np.exp(-gamma * pairwise_sq_dists)

Z = z @ z.T
K = rbf_kernel(X_normalized, X_normalized)
C = 0.01

def semiparametric_svr(K, Z, y, C=1.0):
    N = K.shape[0]
    V = np.ones((N, N))

    # Define matrices for the quadratic program
    P = cvxopt.matrix(0.5 * (K + Z + V + np.eye(N) / C))
    q = cvxopt.matrix(-y.reshape(-1, 1))

    # Solve the quadratic program
    solution = cvxopt.solvers.qp(P=P, q=q)
    alpha = np.array(solution['x']).flatten()
    return alpha

alpha = semiparametric_svr(K, Z, y, C=1.0)

def compute_parameters(X, z, y, alpha):
    w = np.sum(alpha[:, None] * X, axis=0)
    beta = np.sum(alpha[:, None] * z, axis=0)
    b = np.mean(y - (w @ X.T + beta @ z.T))
    return w, b, beta

w, b, beta = compute_parameters(X, z, y, alpha)

def predict(X_new, w, b, beta, z_new):
    return X_new @ w + z_new @ beta + b

y_pred = predict(X_normalized, w, b, beta, z)
np.sqrt(np.mean((y_pred - y)**2))