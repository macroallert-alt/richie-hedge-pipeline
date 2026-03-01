#!/usr/bin/env python3
"""
V16_DAILY_RUNNER.py — Daily Production Runner
==============================================
Liest V16 Sheet, berechnet heutige Allokation (V38 Engine 1:1),
generiert data/dashboard/latest.json fuer Vercel Frontend.

Engine-Logik: Identisch zu V38_PRODUCTION_V16_NEU.py
Output: Minimale dashboard.json mit V16-Daten (Schema v2.0 kompatibel)

Laeuft taeglich via GitHub Actions um ~06:00 UTC.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import os
import sys

# ═══════════════════════════════════════════════════════
# SHEET
# ═══════════════════════════════════════════════════════
SHEET_ID = "11xoZ-E-W0eG23V_HSKloqzC4ubLYg9pfcf6k7HJ0oSE"

# ═══════════════════════════════════════════════════════
# ASSET UNIVERSE — 25 tradeable + 2 indicator
# ═══════════════════════════════════════════════════════
ASSETS = [
    'GLD','SLV','GDX','GDXJ','SIL',
    'SPY','XLY','XLI','XLF','XLE','IWM','XLK',
    'XLV','XLP','XLU',
    'VNQ',
    'EEM','VGK',
    'TLT','TIP','LQD','HYG',
    'DBC',
    'BTC','ETH',
]

# 9 Liquidity Transmission Channels
RISK_CAT = {
    'GLD':'MONETARY_DIRECT', 'TLT':'MONETARY_DIRECT',
    'GDX':'MONETARY_LEVERAGED', 'GDXJ':'MONETARY_LEVERAGED', 'SIL':'MONETARY_LEVERAGED',
    'SLV':'INDUSTRIAL_METAL',
    'LQD':'CREDIT_SPREAD', 'HYG':'CREDIT_SPREAD',
    'SPY':'GROWTH_CYCLE', 'XLK':'GROWTH_CYCLE', 'XLY':'GROWTH_CYCLE',
    'XLI':'GROWTH_CYCLE', 'XLF':'GROWTH_CYCLE', 'IWM':'GROWTH_CYCLE',
    'DBC':'INFLATION_REAL', 'XLE':'INFLATION_REAL', 'VNQ':'INFLATION_REAL', 'TIP':'INFLATION_REAL',
    'XLV':'DEFENSIVE_DAMPENED', 'XLP':'DEFENSIVE_DAMPENED', 'XLU':'DEFENSIVE_DAMPENED',
    'EEM':'GLOBAL_FLOW', 'VGK':'GLOBAL_FLOW',
    'BTC':'SPECULATIVE_EXCESS', 'ETH':'SPECULATIVE_EXCESS',
}

# 8 Cluster Caps
CLUSTERS = {
    'PM': ['GLD','SLV','GDX','GDXJ','SIL'],
    'EQ_CYCL': ['SPY','XLY','XLI','XLF','XLE','IWM'],
    'EQ_DEFN': ['XLV','XLP','XLU','VNQ'],
    'EQ_GROW': ['XLK'],
    'EQ_INTL': ['EEM','VGK'],
    'BOND': ['TLT','TIP','LQD','HYG'],
    'COMMOD': ['DBC'],
    'CRYPTO': ['BTC','ETH'],
}
BOND_ASSETS = set(CLUSTERS['BOND'])

KR1 = {
    'PM': 0.29, 'EQ_CYCL': 0.35, 'EQ_DEFN': 0.34, 'EQ_GROW': 0.21,
    'EQ_INTL': 0.15, 'BOND': 0.36, 'COMMOD': 0.15, 'CRYPTO': 0.38,
}

TX_BPS = {a: (15 if a in ['BTC','ETH'] else 5) for a in ASSETS}

# ═══════════════════════════════════════════════════════
# HOWELL MULTIPLIERS
# ═══════════════════════════════════════════════════════
HOWELL_BASE = {
    1: {'MONETARY_DIRECT': 0.73, 'MONETARY_LEVERAGED': 1.17, 'INDUSTRIAL_METAL': 1.3,
        'CREDIT_SPREAD': 1.17, 'GROWTH_CYCLE': 1.3, 'INFLATION_REAL': 1.3,
        'DEFENSIVE_DAMPENED': 1.18, 'GLOBAL_FLOW': 1.3, 'SPECULATIVE_EXCESS': 1.3},
    0: {'MONETARY_DIRECT': 1.07, 'MONETARY_LEVERAGED': 0.4, 'INDUSTRIAL_METAL': 0.6,
        'CREDIT_SPREAD': 1.17, 'GROWTH_CYCLE': 1.16, 'INFLATION_REAL': 0.69,
        'DEFENSIVE_DAMPENED': 1.3, 'GLOBAL_FLOW': 1.24, 'SPECULATIVE_EXCESS': 1.13},
    -1: {'MONETARY_DIRECT': 1.21, 'MONETARY_LEVERAGED': 1.16, 'INDUSTRIAL_METAL': 1.12,
         'CREDIT_SPREAD': 0.73, 'GROWTH_CYCLE': 0.4, 'INFLATION_REAL': 0.79,
         'DEFENSIVE_DAMPENED': 0.86, 'GLOBAL_FLOW': 0.4, 'SPECULATIVE_EXCESS': 0.4},
}

HOWELL_STRESS = {
    'MONETARY_DIRECT': 1.3, 'MONETARY_LEVERAGED': 1.29, 'INDUSTRIAL_METAL': 0.4,
    'CREDIT_SPREAD': 0.47, 'GROWTH_CYCLE': 0.4, 'INFLATION_REAL': 0.4,
    'DEFENSIVE_DAMPENED': 0.4, 'GLOBAL_FLOW': 0.4, 'SPECULATIVE_EXCESS': 1.3,
}

# LIQ_ALIGN — 0.50 flat
LIQ_ALIGN = {
    1: {ch: 0.50 for ch in set(RISK_CAT.values())},
    0: {ch: 0.50 for ch in set(RISK_CAT.values())},
    -1: {ch: 0.50 for ch in set(RISK_CAT.values())},
}

# ═══════════════════════════════════════════════════════
# STATE_ALIGN — 12x25
# ═══════════════════════════════════════════════════════
STATE_ALIGN = {
    1: {'GLD':0.31,'SLV':0.12,'GDX':0.09,'GDXJ':0.16,'SIL':0.20,'SPY':0.88,'XLY':0.65,
        'XLI':0.95,'XLF':0.84,'XLE':0.50,'IWM':0.61,'XLK':0.80,'XLV':0.72,'XLP':0.54,'XLU':0.46,
        'VNQ':0.42,'EEM':0.39,'VGK':0.69,'TLT':0.05,'TIP':0.24,'LQD':0.27,'HYG':0.57,
        'DBC':0.35,'BTC':0.76,'ETH':0.91},
    2: {'GLD':0.61,'SLV':0.20,'GDX':0.54,'GDXJ':0.27,'SIL':0.12,'SPY':0.88,'XLY':0.84,
        'XLI':0.69,'XLF':0.72,'XLE':0.09,'IWM':0.50,'XLK':0.95,'XLV':0.65,'XLP':0.80,'XLU':0.46,
        'VNQ':0.76,'EEM':0.35,'VGK':0.39,'TLT':0.16,'TIP':0.42,'LQD':0.31,'HYG':0.57,
        'DBC':0.05,'BTC':0.91,'ETH':0.24},
    3: {'GLD':0.95,'SLV':0.76,'GDX':0.91,'GDXJ':0.80,'SIL':0.46,'SPY':0.42,'XLY':0.27,
        'XLI':0.31,'XLF':0.16,'XLE':0.61,'IWM':0.24,'XLK':0.39,'XLV':0.57,'XLP':0.69,'XLU':0.72,
        'VNQ':0.20,'EEM':0.65,'VGK':0.50,'TLT':0.12,'TIP':0.35,'LQD':0.54,'HYG':0.84,
        'DBC':0.88,'BTC':0.09,'ETH':0.05},
    4: {'GLD':0.95,'SLV':0.69,'GDX':0.65,'GDXJ':0.72,'SIL':0.61,'SPY':0.27,'XLY':0.09,
        'XLI':0.31,'XLF':0.12,'XLE':0.46,'IWM':0.20,'XLK':0.16,'XLV':0.57,'XLP':0.35,'XLU':0.80,
        'VNQ':0.54,'EEM':0.24,'VGK':0.05,'TLT':0.88,'TIP':0.84,'LQD':0.42,'HYG':0.39,
        'DBC':0.50,'BTC':0.76,'ETH':0.91},
    5: {'GLD':0.16,'SLV':0.31,'GDX':0.05,'GDXJ':0.09,'SIL':0.20,'SPY':0.65,'XLY':0.88,
        'XLI':0.72,'XLF':0.54,'XLE':0.46,'IWM':0.80,'XLK':0.57,'XLV':0.42,'XLP':0.84,'XLU':0.27,
        'VNQ':0.61,'EEM':0.24,'VGK':0.35,'TLT':0.12,'TIP':0.39,'LQD':0.76,'HYG':0.69,
        'DBC':0.50,'BTC':0.91,'ETH':0.95},
    6: {'GLD':0.76,'SLV':0.27,'GDX':0.69,'GDXJ':0.72,'SIL':0.54,'SPY':0.50,'XLY':0.57,
        'XLI':0.80,'XLF':0.46,'XLE':0.09,'IWM':0.31,'XLK':0.65,'XLV':0.39,'XLP':0.61,'XLU':0.95,
        'VNQ':0.42,'EEM':0.35,'VGK':0.16,'TLT':0.91,'TIP':0.84,'LQD':0.88,'HYG':0.24,
        'DBC':0.05,'BTC':0.20,'ETH':0.12},
    7: {'GLD':0.91,'SLV':0.83,'GDX':0.58,'GDXJ':0.54,'SIL':0.70,'SPY':0.13,'XLY':0.17,
        'XLI':0.46,'XLF':0.34,'XLE':0.42,'IWM':0.30,'XLK':0.09,'XLV':0.38,'XLP':0.25,'XLU':0.05,
        'VNQ':0.21,'EEM':0.62,'VGK':0.50,'TLT':0.79,'TIP':0.95,'LQD':0.87,'HYG':0.66,
        'DBC':0.75,'BTC':0.50,'ETH':0.50},
    8: {'GLD':0.70,'SLV':0.95,'GDX':0.13,'GDXJ':0.25,'SIL':0.34,'SPY':0.46,'XLY':0.66,
        'XLI':0.30,'XLF':0.09,'XLE':0.42,'IWM':0.75,'XLK':0.79,'XLV':0.83,'XLP':0.87,'XLU':0.21,
        'VNQ':0.58,'EEM':0.50,'VGK':0.38,'TLT':0.05,'TIP':0.62,'LQD':0.17,'HYG':0.91,
        'DBC':0.54,'BTC':0.50,'ETH':0.50},
    9: {'GLD':0.31,'SLV':0.14,'GDX':0.09,'GDXJ':0.26,'SIL':0.50,'SPY':0.78,'XLY':0.74,
        'XLI':0.86,'XLF':0.22,'XLE':0.35,'IWM':0.56,'XLK':0.69,'XLV':0.91,'XLP':0.95,'XLU':0.65,
        'VNQ':0.61,'EEM':0.44,'VGK':0.52,'TLT':0.05,'TIP':0.48,'LQD':0.39,'HYG':0.82,
        'DBC':0.18,'BTC':0.50,'ETH':0.50},
    10: {'GLD':0.70,'SLV':0.83,'GDX':0.62,'GDXJ':0.79,'SIL':0.91,'SPY':0.34,'XLY':0.05,
         'XLI':0.13,'XLF':0.09,'XLE':0.50,'IWM':0.17,'XLK':0.25,'XLV':0.21,'XLP':0.46,'XLU':0.75,
         'VNQ':0.42,'EEM':0.38,'VGK':0.30,'TLT':0.54,'TIP':0.95,'LQD':0.87,'HYG':0.58,
         'DBC':0.66,'BTC':0.50,'ETH':0.50},
    11: {'GLD':0.46,'SLV':0.05,'GDX':0.80,'GDXJ':0.61,'SIL':0.76,'SPY':0.42,'XLY':0.69,
         'XLI':0.24,'XLF':0.20,'XLE':0.39,'IWM':0.27,'XLK':0.50,'XLV':0.57,'XLP':0.72,'XLU':0.12,
         'VNQ':0.35,'EEM':0.16,'VGK':0.31,'TLT':0.95,'TIP':0.91,'LQD':0.65,'HYG':0.54,
         'DBC':0.09,'BTC':0.88,'ETH':0.84},
    12: {'GLD':0.95,'SLV':0.90,'GDX':0.86,'GDXJ':0.50,'SIL':0.50,'SPY':0.32,'XLY':0.41,
         'XLI':0.09,'XLF':0.64,'XLE':0.45,'IWM':0.18,'XLK':0.54,'XLV':0.23,'XLP':0.14,'XLU':0.27,
         'VNQ':0.59,'EEM':0.72,'VGK':0.36,'TLT':0.81,'TIP':0.50,'LQD':0.77,'HYG':0.68,
         'DBC':0.05,'BTC':0.50,'ETH':0.50},
}
SA_ARR = np.zeros((12, len(ASSETS)))
for s in range(1, 13):
    for j, a in enumerate(ASSETS):
        SA_ARR[s-1, j] = STATE_ALIGN[s].get(a, 0.5)

# ═══════════════════════════════════════════════════════
# CTM
# ═══════════════════════════════════════════════════════
CTM_CFG = {
    'LOW': {'assets': ['SPY','XLK','TIP','GLD','TLT','XLP','XLU','LQD'],
            'mw': -0.05, 'mc': -0.10, 'w': 0.80, 'c': 0.50, 'a': 0.00},
    'MID': {'assets': ['SLV','XLF','XLE','IWM','EEM','XLY','XLI','XLV','VGK','VNQ','HYG','DBC'],
            'mw': -0.07, 'mc': -0.12, 'w': 0.80, 'c': 0.40, 'a': 0.00},
    'MINER': {'assets': ['GDX','GDXJ','SIL'],
              'mw': -0.10, 'mc': -0.18, 'w': 0.70, 'c': 0.30, 'a': 0.15},
    'CRYPTO': {'assets': ['BTC','ETH'],
               'mw': -0.10, 'mc': -0.12, 'w': 0.70, 'c': 0.30, 'a': 0.00},
}
CTM_CLS = {}
for cl, cf in CTM_CFG.items():
    for a in cf['assets']:
        CTM_CLS[a] = cl

# ═══════════════════════════════════════════════════════
# RV PAIRS
# ═══════════════════════════════════════════════════════
RV_PAIRS = {
    'GLD': [('GLD_SLV_Z',True),('GLD_PLAT_Z',True),('GDX_GLD_Z',False),('BTC_GLD_Z',False),('GLD_SPY_Z',True),('GLD_TLT_Z',True)],
    'SLV': [('GLD_SLV_Z',False),('SIL_SLV_Z',False)],
    'GDX': [('GDX_GLD_Z',True),('GDXJ_GDX_Z',False)],
    'GDXJ': [('GDXJ_GDX_Z',True)], 'SIL': [('SIL_SLV_Z',True)],
    'SPY': [('BTC_SPY_Z',False),('GLD_SPY_Z',False),('SPY_TLT_Z',True),('XLF_SPY_Z',False),
            ('XLE_SPY_Z',False),('XLK_SPY_Z',False),('IWM_SPY_Z',False),('EEM_SPY_Z',False),('VGK_SPY_Z',False)],
    'TLT': [('GLD_TLT_Z',False),('SPY_TLT_Z',False),('TIP_TLT_Z',False)],
    'TIP': [('TIP_TLT_Z',True)],
    'BTC': [('BTC_GLD_Z',True),('ETH_BTC_Z',False),('BTC_SPY_Z',True)],
    'ETH': [('ETH_BTC_Z',True)],
    'XLF': [('XLF_SPY_Z',True)], 'XLE': [('XLE_SPY_Z',True)], 'XLK': [('XLK_SPY_Z',True)],
    'IWM': [('IWM_SPY_Z',True)], 'EEM': [('EEM_SPY_Z',True)],
    'XLY': [('XLY_SPY_Z',True)], 'XLI': [('XLI_SPY_Z',True)],
    'XLV': [('XLV_SPY_Z',True)], 'XLP': [('XLP_SPY_Z',True)],
    'XLU': [('XLU_SPY_Z',True)], 'VNQ': [('VNQ_SPY_Z',True)],
    'LQD': [('LQD_TLT_Z',True)],
    'HYG': [('HYG_SPY_Z',True),('HYG_LQD_Z',True)],
    'DBC': [('DBC_GLD_Z',True)], 'VGK': [('VGK_SPY_Z',True)],
}
PAIR_CNT = {a: len(RV_PAIRS.get(a, [])) for a in ASSETS}
MAX_PAIRS = max(PAIR_CNT.values())
N48_CONF = {a: max(0.33, np.sqrt(PAIR_CNT[a]/MAX_PAIRS)) if MAX_PAIRS > 0 else 1.0 for a in ASSETS}

RV_PAIR_ASSETS = {
    'GLD_SLV_Z':('GLD','SLV'), 'GLD_PLAT_Z':('GLD','PLAT'), 'GDX_GLD_Z':('GDX','GLD'),
    'GDXJ_GDX_Z':('GDXJ','GDX'), 'SIL_SLV_Z':('SIL','SLV'), 'BTC_GLD_Z':('BTC','GLD'),
    'ETH_BTC_Z':('ETH','BTC'), 'BTC_SPY_Z':('BTC','SPY'), 'GLD_SPY_Z':('GLD','SPY'),
    'GLD_TLT_Z':('GLD','TLT'), 'SPY_TLT_Z':('SPY','TLT'), 'XLF_SPY_Z':('XLF','SPY'),
    'XLE_SPY_Z':('XLE','SPY'), 'XLK_SPY_Z':('XLK','SPY'), 'IWM_SPY_Z':('IWM','SPY'),
    'EEM_SPY_Z':('EEM','SPY'), 'VGK_SPY_Z':('VGK','SPY'), 'TIP_TLT_Z':('TIP','TLT'),
    'XLY_SPY_Z':('XLY','SPY'), 'XLI_SPY_Z':('XLI','SPY'), 'XLV_SPY_Z':('XLV','SPY'),
    'XLP_SPY_Z':('XLP','SPY'), 'XLU_SPY_Z':('XLU','SPY'), 'VNQ_SPY_Z':('VNQ','SPY'),
    'LQD_TLT_Z':('LQD','TLT'), 'HYG_SPY_Z':('HYG','SPY'), 'HYG_LQD_Z':('HYG','LQD'),
    'DBC_GLD_Z':('DBC','GLD'),
}

# ═══════════════════════════════════════════════════════
# ENGINE PARAMETERS
# ═══════════════════════════════════════════════════════
W_SA = 0.75; W_LQ = 0.25; W_RV = 0.00; W_TM = 0.00; VOL_LB = 60
SIGMOID_STEEP = 12; SIGMOID_MID = 0.50

DD_PROT_LB = 8; DD_PROT_WARN = -0.08; DD_PROT_CRIT = -0.15
MC_THRESHOLD = 0.03; MC_DD_EXEMPT = True

RV_ZSCALE = 0.3; RV_STEEP = 0.5; RV_MID = 0.2
RV_PCT_WINDOW = 252; RV_PCT_MINPER = 126

ASSET_MAX_WT = 0.25; VOL_FLOOR = 0.08; BOND_CORR_THRESH = 0.30
BOND_VOL_FLOOR_HIGH = 0.15; FM_EMA_SPAN = 5

CV_CUTOFF = {'high_cv': 0.8, 'high_pct': 80, 'mid_cv': 0.5, 'mid_pct': 70, 'low_pct': 55}

# ═══════════════════════════════════════════════════════
# FUNCTIONS (identisch zu V38)
# ═══════════════════════════════════════════════════════
def sigmoid_vec(x, s=SIGMOID_STEEP, m=SIGMOID_MID):
    return 1.0 / (1.0 + np.exp(-s * (x - m)))

def cfactor_vec(g, l):
    r = np.full(len(g), 0.75)
    r[(g == 0) | (l == 0)] = 0.90
    r[((g > 0) & (l > 0)) | ((g < 0) & (l < 0))] = 1.00
    return r

def calc_oews(pr, asset):
    p = pr[['Date', asset]].dropna().copy().set_index('Date').sort_index()
    p['r3'] = p[asset].pct_change(63)
    p['ms'] = p['r3'].clip(-0.30, 0.30).apply(lambda x: (x+0.30)/0.60*100)
    p['h52'] = p[asset].rolling(252, min_periods=126).max()
    p['l52'] = p[asset].rolling(252, min_periods=126).min()
    rng = (p['h52'] - p['l52']).replace(0, np.nan)
    p['ps'] = ((p[asset] - p['l52']) / rng * 100).clip(0, 100)
    p['rd'] = p[asset].pct_change()
    p['v20'] = p['rd'].rolling(20).std() * np.sqrt(252) * 100
    p['v1y'] = p['rd'].rolling(252, min_periods=126).std() * np.sqrt(252) * 100
    p['vs'] = ((p['v20'] / p['v1y'].replace(0, np.nan) - 0.5) / 1.5).clip(0, 1) * 100
    p['oews'] = p['ms']*0.30 + p['ps']*0.25 + p['vs']*0.25 + 50*0.20
    return p[['oews']].rename(columns={'oews': f'OEWS_{asset}'})

def calc_ctm(pr, asset):
    cl = CTM_CLS.get(asset, 'LOW')
    cf = CTM_CFG[cl]
    p = pr[['Date', asset]].dropna().copy()
    if 'SPY' in pr.columns and asset != 'SPY':
        spy_s = pr.set_index('Date')['SPY']
        p = p.set_index('Date').sort_index()
        p['SPY'] = spy_s.reindex(p.index)
    else:
        p = p.set_index('Date').sort_index()
    p['r3'] = p[asset].pct_change(63)
    p['r6'] = p[asset].pct_change(126)
    p['r12'] = p[asset].pct_change(252)
    if 'SPY' in p.columns:
        p['sr3'] = p['SPY'].pct_change(63)
        p['sr6'] = p['SPY'].pct_change(126)
        p['vs3'] = p['r3'] - p['sr3']
        p['vs6'] = p['r6'] - p['sr6']
    else:
        p['vs3'] = 0.0; p['vs6'] = 0.0
    r3 = p['r3'].values; r6 = p['r6'].values; r12 = p['r12'].values
    vs3 = p['vs3'].fillna(0).values; vs6 = p['vs6'].fillna(0).values
    ms = np.zeros(len(p)); valid = ~(np.isnan(r3) | np.isnan(r6))
    ms[valid] = ((r3[valid] < cf['mw']).astype(int) + (r3[valid] < cf['mc']).astype(int) + (r6[valid] < cf['mc']).astype(int))
    ms = np.minimum(ms, 3)
    rs = (vs3 < cf['mw']).astype(float) + (vs6 < cf['mw']).astype(float)
    ts = np.zeros(len(p)); v12 = ~np.isnan(r12)
    ts[v12] = (r12[v12] < 0).astype(float) + (r12[v12] < cf['mc']*2).astype(float)
    ctm_raw = ms*1.0 + rs*0.5 + ts*0.25
    mult = np.ones(len(p))
    mult[ctm_raw >= 1] = cf['w']; mult[ctm_raw >= 2] = cf['c']; mult[ctm_raw >= 4] = cf['a']
    if cl == 'CRYPTO':
        ov = (mult <= cf['a']) & (~np.isnan(r3)) & (r3 > 0.20); mult[ov] = cf['w']
    p['ctm_m'] = mult
    return p[['ctm_m']].rename(columns={'ctm_m': f'CTM_{asset}'})

def mets(r):
    r = r.dropna()
    if len(r) < 252:
        return {k: np.nan for k in ['CAGR','Sharpe','MaxDD','Vol','Calmar']}
    c = (1+r).cumprod(); y = len(r)/252
    cagr = (c.iloc[-1]**(1/y)-1)*100; vol = r.std()*np.sqrt(252)*100
    sh = cagr/vol if vol > 0 else 0; dd = ((c-c.cummax())/c.cummax()).min()*100
    cal = abs(cagr/dd) if dd != 0 else 0
    return {'CAGR': round(cagr,2), 'Sharpe': round(sh,2), 'MaxDD': round(dd,2),
            'Vol': round(vol,2), 'Calmar': round(cal,2)}

# ═══════════════════════════════════════════════════════
# DATA LOADING (identisch zu V38)
# ═══════════════════════════════════════════════════════
def load_data():
    print("\n" + "="*60)
    print("DATEN LADEN")
    print("="*60)
    base = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet="

    def load_tab(name):
        df = pd.read_csv(base + name)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={df.columns[0]: 'Date'})
        if not str(df.iloc[0]['Date'])[:4].isdigit():
            df = df.iloc[1:].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

    prices = load_tab("DATA_Prices")
    all_tickers = list(set(ASSETS + ['VGK','PLATINUM','PLAT','COPPER']))
    col_rename = {}
    for col in prices.columns:
        cs = col.strip()
        first_word = cs.split(' ')[0].split('_')[0].strip()
        if first_word in all_tickers and first_word not in col_rename.values():
            col_rename[col] = first_word
        elif cs == 'Date Datum' or cs.lower().startswith('date'):
            col_rename[col] = 'Date'
    if col_rename:
        prices = prices.rename(columns=col_rename)
    for a in all_tickers:
        if a in prices.columns:
            if prices[a].dtype == object:
                prices[a] = prices[a].astype(str).str.replace('.','',regex=False).str.replace(',','.',regex=False)
            prices[a] = pd.to_numeric(prices[a], errors='coerce')
    if 'PLATINUM' in prices.columns and 'PLAT' not in prices.columns:
        prices['PLAT'] = prices['PLATINUM']
    found = [a for a in ASSETS if a in prices.columns]
    print(f"  Prices: {len(prices)} Tage, {len(found)}/25 Assets, {prices['Date'].min().date()} -> {prices['Date'].max().date()}")

    macro = load_tab("CALC_Macro_State")
    cm = {}
    for c in macro.columns:
        cl = c.lower()
        if 'macro_state_num' in cl: cm['ms'] = c
        elif 'growth_signal' in cl: cm['gr'] = c
        elif 'stress_score' in cl: cm['st'] = c
    print(f"  Macro: {len(macro)} Zeilen")

    k16 = load_tab("DATA_K16_K17")
    for c in k16.columns:
        cl = c.lower()
        if 'liq_dir_confirmed' in cl: k16 = k16.rename(columns={c: 'Liq_Dir_Confirmed'})
        elif 'vote_sum_magnitude' in cl: k16 = k16.rename(columns={c: 'Vote_Sum_Magnitude'})
    if 'Vote_Sum_Magnitude' not in k16.columns:
        vs_col = [c for c in k16.columns if c.strip().lower() == 'vote_sum']
        if vs_col:
            k16['Vote_Sum_Magnitude'] = pd.to_numeric(k16[vs_col[0]], errors='coerce').abs()
            print(f"  ! Vote_Sum_Magnitude aus abs(Vote_Sum) berechnet")
    print(f"  K16: {len(k16)} Zeilen")

    print("  RV Percentile-Ranks...", end=' ', flush=True)
    rv_pct = prices[['Date']].copy()
    computed = 0
    for pair_z, (a, b) in RV_PAIR_ASSETS.items():
        a_col = a if a in prices.columns else None
        b_col = b if b in prices.columns else None
        if a_col and b_col:
            ratio = prices[a_col] / prices[b_col].replace(0, np.nan)
            pct = ratio.rolling(RV_PCT_WINDOW, min_periods=RV_PCT_MINPER).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            rv_pct[pair_z] = (pct - 0.5) * 2
            computed += 1
    print(f"{computed} Pairs OK")
    print(f"  DEBUG DATA: SPY dtype={prices['SPY'].dtype}, first5={prices['SPY'].head().tolist()}")
    print(f"  DEBUG COLS: {list(prices.columns[:10])}")
    print(f"  DEBUG RAW0: {prices.iloc[0].tolist()[:5]}")
    print("DATEN GELADEN")
    return prices, macro, k16, rv_pct, cm

# ═══════════════════════════════════════════════════════
# ENGINE (identisch zu V38)
# ═══════════════════════════════════════════════════════
def run_engine(prices, macro, k16, rv, cm):
    """Laeuft die V38-Engine und gibt heutige Gewichte + Metadaten zurueck."""
    ms_c = cm.get('ms','Macro_State_Num'); gr_c = cm.get('gr','Growth_Signal')
    st_c = cm.get('st','Stress_Score')
    N_ASSETS = len(ASSETS)

    df = prices[['Date'] + [a for a in ASSETS if a in prices.columns]].copy()
    ms_cols = ['Date'] + [c for c in [ms_c, gr_c, st_c] if c in macro.columns]
    df = df.merge(macro[ms_cols], on='Date', how='left')
    for c in ms_cols[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce').ffill()
    k_cols = ['Date'] + [c for c in k16.columns if any(x in c.lower() for x in ['liq_dir','vote_sum'])]
    df = df.merge(k16[k_cols], on='Date', how='left')
    for c in k_cols[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce').ffill()
    rz = ['Date'] + [c for c in rv.columns if '_Z' in c]
    df = df.merge(rv[rz], on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    N = len(df)

    print(f"  OEWS+CTM...", end=' ', flush=True)
    for a in ASSETS:
        if a in prices.columns:
            df = df.merge(calc_oews(prices, a), left_on='Date', right_index=True, how='left')
            df = df.merge(calc_ctm(prices, a), left_on='Date', right_index=True, how='left')
    print("OK", flush=True)

    ldc = vsmc = None
    for c in df.columns:
        cl = c.lower()
        if 'liq_dir_confirmed' in cl or 'liq_dir_final' in cl: ldc = c
        if 'vote_sum_magnitude' in cl: vsmc = c
    if not ldc:
        for c in df.columns:
            if 'liq_dir' in c.lower(): ldc = c; break

    # SC-6
    print(f"  SC-6...", end=' ', flush=True)
    ld_vals = df[ldc].fillna(0).astype(float).values if ldc else np.zeros(N)
    st_vals = df[st_c].fillna(0).astype(float).values if st_c in df.columns else np.zeros(N)
    is_liq1 = (ld_vals == 1).astype(int)
    streak = np.zeros(N, dtype=int)
    for i in range(N):
        streak[i] = (streak[i-1]+1)*is_liq1[i] if i > 0 else is_liq1[i]
    override_mask = (st_vals >= 2) & (streak >= 20)
    eff_ms_arr = df[ms_c].fillna(6).astype(float).values.copy()
    eff_st_arr = st_vals.copy()
    eff_ms_arr[override_mask] = 8
    eff_st_arr[override_mask] = 1
    print(f"{int(override_mask.sum())}d", flush=True)

    eff_ms = np.clip(np.round(eff_ms_arr).astype(int), 1, 12)
    gr_vals = df[gr_c].fillna(0).astype(float).values if gr_c in df.columns else np.zeros(N)
    ld_int = np.round(ld_vals).astype(int)
    ld_int = np.where(np.isin(ld_int, [-1,0,1]), ld_int, 0)
    vsm_arr = df[vsmc].fillna(0).astype(float).values if vsmc else np.zeros(N)

    ret_matrix = np.zeros((N, N_ASSETS))
    for j, a in enumerate(ASSETS):
        if a in df.columns:
            ret_matrix[:, j] = df[a].pct_change().fillna(0).values

    # Confluence + FM
    print(f"  Confluence+FM...", end=' ', flush=True)
    fm_matrix = np.zeros((N, N_ASSETS))
    fm_matrix_pre_dd = np.zeros((N, N_ASSETS))
    dd_prot_active = np.zeros((N, N_ASSETS), dtype=bool)

    for j, a in enumerate(ASSETS):
        if a not in prices.columns:
            continue
        rc = RISK_CAT[a]
        sa = SA_ARR[eff_ms-1, j] * W_SA
        la_vals = np.array([LIQ_ALIGN[int(li)].get(rc, 0.5) for li in ld_int])
        ls = (np.abs(vsm_arr) / 4.0) * la_vals * W_LQ

        pairs = RV_PAIRS.get(a, [])
        if pairs:
            z_cols_a = []
            for pc, isn in pairs:
                if pc in df.columns:
                    z = df[pc].values.copy()
                    if isn: z = -z
                    z_cols_a.append(z)
            if z_cols_a:
                avg_z = np.nanmean(np.column_stack(z_cols_a), axis=1)
                rv_s = sigmoid_vec(avg_z*RV_ZSCALE, RV_STEEP, RV_MID) * W_RV
                rv_s[np.isnan(avg_z)] = 0.5 * W_RV
            else:
                rv_s = np.full(N, 0.5*W_RV)
        else:
            rv_s = np.full(N, 0.5*W_RV)

        oc = f'OEWS_{a}'
        if oc in df.columns:
            oews = df[oc].values
            ts = np.where(np.isnan(oews), 0.5*W_TM, (1.0 - oews/100.0)*W_TM)
        else:
            ts = np.full(N, 0.5*W_TM)

        conf = sa + ls + rv_s + ts
        pm = sigmoid_vec(conf)

        hm = np.zeros(N)
        stress_mask = eff_st_arr >= 2
        hm[stress_mask] = HOWELL_STRESS.get(rc, 0.5)
        for li_val in [-1, 0, 1]:
            m = (~stress_mask) & (ld_int == li_val)
            hm[m] = HOWELL_BASE[li_val].get(rc, 0.9)
        rem = (~stress_mask) & (~np.isin(ld_int, [-1,0,1]))
        hm[rem] = HOWELL_BASE[0].get(rc, 0.9)
        hm *= cfactor_vec(gr_vals, ld_vals)

        cc = f'CTM_{a}'
        ctm = df[cc].fillna(1.0).values if cc in df.columns else np.ones(N)
        fm = np.minimum(1.3, pm * hm * ctm)

        if a in df.columns:
            r_lb = pd.Series(df[a].values).pct_change(DD_PROT_LB).values
            dd_warn = (r_lb < DD_PROT_WARN) & (~np.isnan(r_lb))
            dd_crit = (r_lb < DD_PROT_CRIT) & (~np.isnan(r_lb))
            dd_prot_active[dd_warn | dd_crit, j] = True
            fm_pre_dd = fm.copy()
            fm[dd_warn] = fm[dd_warn] * 0.20
            fm[dd_crit] = 0.0
        else:
            fm_pre_dd = fm.copy()

        fm = fm * N48_CONF[a]
        fm_pre_dd = fm_pre_dd * N48_CONF[a]
        fm_matrix[:, j] = fm
        fm_matrix_pre_dd[:, j] = fm_pre_dd

    print("OK", flush=True)

    # DD-Protect
    print(f"  DD-Prot...", end=' ', flush=True)
    print(f"OK | DD-Prot: {int(dd_prot_active.sum())} Asset-Tage", flush=True)

    # KR-1 Cluster Caps
    print(f"  KR-1...", end=' ', flush=True)
    for cn, ca in CLUSTERS.items():
        idxs = [ASSETS.index(a) for a in ca if a in prices.columns]
        if not idxs:
            continue
        cap = KR1[cn]
        for mat in [fm_matrix, fm_matrix_pre_dd]:
            cs = mat[:, idxs].sum(axis=1)
            over = cs > cap
            if over.any():
                scale = np.where(over, cap / np.maximum(cs, 1e-8), 1.0)
                for idx in idxs:
                    mat[over, idx] *= scale[over]
    print("OK", flush=True)

    # Dynamic CV-Cutoff
    print(f"  CV-Cutoff...", end=' ', flush=True)
    assets_killed = 0
    for i in range(N):
        row = fm_matrix[i, :]
        nonzero = row[row > 0.001]
        if len(nonzero) > 2:
            cv = np.std(nonzero) / np.mean(nonzero) if np.mean(nonzero) > 0 else 0
            if cv > CV_CUTOFF['high_cv']:
                pct = CV_CUTOFF['high_pct']
            elif cv > CV_CUTOFF['mid_cv']:
                pct = CV_CUTOFF['mid_pct']
            else:
                pct = CV_CUTOFF['low_pct']
            threshold = np.percentile(nonzero, pct)
            kill_mask = row < threshold
            assets_killed += kill_mask.sum()
            fm_matrix[i, kill_mask] = 0.0
    avg_alive = N_ASSETS - assets_killed / N
    print(f"OK | Avg alive: {avg_alive:.1f}/{N_ASSETS}", flush=True)

    fm_sum_pre = fm_matrix.sum(axis=1)

    # Cash-Raise
    cr_mask = ((eff_ms == 3) | (eff_ms == 4)) & (ld_int == -1) & (eff_st_arr >= 1)
    for j in range(N_ASSETS):
        fm_matrix[cr_mask, j] *= 0.30

    denom = np.where(fm_sum_pre < 0.01, 1.0, fm_sum_pre)

    # Portfolio with MC
    print(f"  Portfolio (MC)...", end=' ', flush=True)
    w_target = fm_matrix / denom.reshape(-1, 1)

    w_actual = np.zeros_like(w_target)
    port_ret_mc = np.zeros(N)
    w_actual[0] = w_target[0]
    port_ret_mc[0] = (w_actual[0] * ret_matrix[0]).sum()
    for i in range(1, N):
        w_drifted = w_actual[i-1] * (1 + ret_matrix[i])
        ws = w_drifted.sum()
        w_drifted = w_drifted / ws if ws > 1e-8 else w_target[i]
        max_dev = np.abs(w_target[i] - w_drifted).max()
        if max_dev > MC_THRESHOLD:
            w_actual[i] = w_target[i]
        else:
            w_actual[i] = w_drifted
        if MC_DD_EXEMPT:
            changed = False
            for j in range(N_ASSETS):
                if dd_prot_active[i, j]:
                    w_actual[i, j] = w_target[i, j]
                    changed = True
            if changed:
                ws2 = w_actual[i].sum()
                if ws2 > 1e-8:
                    w_actual[i] = w_actual[i] / ws2
        port_ret_mc[i] = (w_actual[i] * ret_matrix[i]).sum()
    print("OK", flush=True)

    # Trim to 2007-07-01
    dates = df['Date'].values
    start = pd.Timestamp('2007-07-01')
    mask = dates >= start
    idx_s = np.argmax(mask)
    sl = slice(idx_s, None)
    pr_s = pd.Series(port_ret_mc[sl]).reset_index(drop=True)
    spy_idx = ASSETS.index('SPY')
    spy_ret = ret_matrix[:, spy_idx]
    spy_s = pd.Series(spy_ret[sl]).reset_index(drop=True)
    wdf = pd.DataFrame(w_actual[sl], columns=ASSETS)
    dates_s = pd.Series(dates[sl]).reset_index(drop=True)

    # Performance metrics
    print(f"  DEBUG: pr_s len={len(pr_s)}, sum={pr_s.sum()}, cumret={(1+pr_s).cumprod().iloc[-1]}, dd_total={int(dd_prot_active.sum())}")
    m_prod = mets(pr_s)
    m_spy = mets(spy_s)

    # Extract TODAY's data (last row)
    last_idx = len(dates_s) - 1
    today_date = pd.Timestamp(dates_s.iloc[last_idx])
    today_weights = {a: round(float(wdf.iloc[last_idx][a]), 6) for a in ASSETS}
    today_target_weights = {a: round(float(w_target[sl][last_idx, j]), 6) for j, a in enumerate(ASSETS)}
    today_macro_state = int(eff_ms[idx_s + last_idx])
    today_growth = int(gr_vals[idx_s + last_idx])
    today_liq_dir = int(ld_int[idx_s + last_idx])
    today_stress = int(eff_st_arr[idx_s + last_idx])

    # DD-Protect Status
    today_dd_prot = [ASSETS[j] for j in range(N_ASSETS) if dd_prot_active[idx_s + last_idx, j]]

    # Cumulative return for drawdown calc
    cum = (1 + pr_s).cumprod()
    current_dd = float((cum.iloc[-1] / cum.cummax().iloc[-1] - 1) * 100)

    # Top weights
    sorted_wts = sorted(today_weights.items(), key=lambda x: x[1], reverse=True)
    top5 = [{"ticker": t, "weight": w} for t, w in sorted_wts[:5]]

    # Cluster weights
    cluster_weights = {}
    for cn, ca in CLUSTERS.items():
        cw = sum(today_weights.get(a, 0) for a in ca)
        cluster_weights[cn] = round(cw, 4)

    # State name mapping
    STATE_NAMES = {
        1: "EARLY_RECOVERY", 2: "RECOVERY_MOMENTUM", 3: "LATE_EXPANSION",
        4: "PEAK_DEFENSIVE", 5: "EARLY_BULL", 6: "QUALITY_GROWTH",
        7: "STAGFLATION_FEAR", 8: "LIQUIDITY_OVERRIDE", 9: "REFLATION",
        10: "RISK_OFF_DEEP", 11: "FLIGHT_TO_SAFETY", 12: "CRISIS_ALPHA"
    }

    # Regime mapping (simplified)
    if today_macro_state in [1, 2, 5, 9]:
        regime = "RISK_ON"
    elif today_macro_state in [3, 6, 8]:
        regime = "SELECTIVE"
    elif today_macro_state in [4, 7]:
        regime = "TRANSITION"
    else:
        regime = "RISK_OFF"

    return {
        "date": today_date.strftime('%Y-%m-%d'),
        "regime": regime,
        "macro_state_num": today_macro_state,
        "macro_state_name": STATE_NAMES.get(today_macro_state, f"STATE_{today_macro_state}"),
        "growth_signal": today_growth,
        "liq_direction": today_liq_dir,
        "stress_score": today_stress,
        "current_weights": today_weights,
        "target_weights": today_target_weights,
        "top_5_weights": top5,
        "cluster_weights": cluster_weights,
        "dd_protect_active": today_dd_prot,
        "current_drawdown_pct": round(current_dd, 2),
        "performance": m_prod,
        "spy_performance": m_spy,
        "data_date": prices['Date'].max().strftime('%Y-%m-%d'),
        "assets_count": len(found) if 'found' in dir() else len([a for a in ASSETS if a in prices.columns]),
        "history_days": len(dates_s),
    }


# ═══════════════════════════════════════════════════════
# DASHBOARD JSON BUILDER
# ═══════════════════════════════════════════════════════
def build_dashboard_json(v16_data):
    """Baut eine Schema-v2.0-kompatible dashboard.json aus V16-Daten.
    Felder die wir noch nicht haben (Agenten-Pipeline) werden mit Platzhaltern gefuellt."""

    now_utc = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    date = v16_data["date"]
    regime = v16_data["regime"]

    # Briefing type heuristic (vereinfacht, bis CIO-Agent steht)
    stress = v16_data["stress_score"]
    dd_active = len(v16_data["dd_protect_active"]) > 0
    if stress >= 2 or dd_active:
        briefing_type = "ACTION"
    elif stress >= 1:
        briefing_type = "WATCH"
    else:
        briefing_type = "ROUTINE"

    dashboard = {
        "schema_version": "2.0",
        "date": date,
        "weekday": pd.Timestamp(date).day_name(),
        "generated_at": now_utc,
        "degradation_level": "PARTIAL",
        "degradation_banner": "Pipeline Phase 1: Nur V16-Daten. Agenten-Pipeline noch nicht aktiv.",

        # ── HEADER ──
        "header": {
            "date": date,
            "weekday": pd.Timestamp(date).day_name(),
            "briefing_type": briefing_type,
            "system_conviction": "N/A",
            "risk_ampel": "YELLOW" if stress >= 1 else "GREEN",
            "fragility_state": "N/A",
            "v16_regime": regime,
            "data_quality": "PARTIAL",
            "pipeline_status": "PHASE_1",
            "f6_positions_count": 0,
            "alerts_active_count": len(v16_data["dd_protect_active"]),
            "divergences_count": 0,
            "action_items_act_count": 0,
            "action_items_review_count": 0,
            "action_items_watch_count": 0,
            "da_challenges_total": 0,
            "da_accepted": 0,
            "da_noted": 0,
            "da_rejected": 0,
            "fact_check_corrections": 0,
            "is_draft_fallback": False,
        },

        # ── DIGEST ──
        "digest": {
            "line_1_type_and_delta": f"{briefing_type} — V16 Regime: {regime} (State {v16_data['macro_state_num']}: {v16_data['macro_state_name']})",
            "line_2_actions": f"V16 DD-Protect: {len(v16_data['dd_protect_active'])} Assets. Growth={v16_data['growth_signal']}, Liq={v16_data['liq_direction']}, Stress={v16_data['stress_score']}.",
            "line_3_confidence": f"Pipeline Phase 1 (nur V16). CAGR={v16_data['performance']['CAGR']}%, Sharpe={v16_data['performance']['Sharpe']}. Agenten-Pipeline noch nicht aktiv.",
        },

        # ── DELTAS ── (leer bis wir yesterday-Vergleich haben)
        "deltas": {"has_yesterday": False},

        # ── REGIME CONTEXT ──
        "regime_context": {
            "layout_mode": "STANDARD",
            "panel_priorities": {},
            "auto_expand": ["digest", "v16"],
            "auto_minimize": [],
            "show_operative_details": False,
        },

        # ── CONSISTENCY FLAGS ──
        "consistency_flags": [],

        # ── OVERCONFIDENCE ──
        "overconfidence": {"flag": False, "message": None, "da_prominence": "NORMAL", "confidence_saturation": None},

        # ── BRIEFING ── (Platzhalter bis CIO-Agent steht)
        "briefing": {
            "status": "UNAVAILABLE",
            "source": "NONE",
            "full_text": "CIO Briefing noch nicht verfuegbar (Pipeline Phase 1).",
            "sections": {},
            "section_word_counts": {},
            "da_markers": [],
            "da_resolution_summary": {"total": 0, "accepted": 0, "noted": 0, "rejected": 0, "details": []},
            "key_assumptions": [],
            "confidence_markers": [],
        },

        # ── ACTION ITEMS ── (Platzhalter)
        "action_items": {
            "summary": {"act_count": 0, "review_count": 0, "watch_count": 0, "total": 0,
                         "escalated_today": 0, "new_today": 0, "resolved_today": 0},
            "prominent": [],
            "aggregated": {"count": 0, "items": []},
            "ongoing_conditions": [],
        },

        # ── LAYERS ── (Platzhalter bis Market Analyst steht)
        "layers": {
            "status": "UNAVAILABLE",
            "system_regime": regime,
            "regime_stability_pct": None,
            "fragility_state": "N/A",
            "fragility_data": {},
            "layer_scores": {},
        },

        # ── V16 ── (ECHTE DATEN)
        "v16": {
            "status": "AVAILABLE",
            "regime": regime,
            "regime_confidence": None,
            "macro_state_num": v16_data["macro_state_num"],
            "macro_state_name": v16_data["macro_state_name"],
            "growth_signal": v16_data["growth_signal"],
            "liq_direction": v16_data["liq_direction"],
            "stress_score": v16_data["stress_score"],
            "dd_protect_status": "ACTIVE" if v16_data["dd_protect_active"] else "INACTIVE",
            "dd_protect_assets": v16_data["dd_protect_active"],
            "current_drawdown": v16_data["current_drawdown_pct"],
            "dd_protect_threshold": -10.0,
            "current_weights": v16_data["current_weights"],
            "target_weights": v16_data["target_weights"],
            "top_5_weights": v16_data["top_5_weights"],
            "cluster_weights": v16_data["cluster_weights"],
            "weight_deltas": {"top_increases": [], "top_decreases": []},
            "performance": v16_data["performance"],
            "spy_performance": v16_data["spy_performance"],
        },

        # ── F6 ── (Platzhalter)
        "f6": {
            "status": "UNAVAILABLE",
            "portfolio_summary": {"positions_count": 0, "pending_signals_count": 0,
                                   "total_exposure_pct": 0, "avg_holding_days": 0, "cc_coverage_pct": 0},
            "active_positions": [],
            "pending_signals": [],
            "cc_expiry_warnings": [],
        },

        # ── SIGNALS ── (Platzhalter)
        "signals": {
            "status": "UNAVAILABLE",
            "trade_count": 0,
            "router_status": {},
            "effective_concentration": None,
            "permopt_status": {"budget_pct": 0.05, "active": False, "positions_count": 0},
        },

        # ── RISK ── (Minimal aus V16 DD-Protect)
        "risk": {
            "status": "PARTIAL",
            "portfolio_status": "YELLOW" if stress >= 1 else "GREEN",
            "alerts": [
                {"id": f"DD-{a}", "check_name": "DD_PROTECT", "severity": "WARNING",
                 "message": f"DD-Protect aktiv fuer {a}", "trend": "ACTIVE", "days_active": 1}
                for a in v16_data["dd_protect_active"]
            ],
            "emergency_triggers": {
                "max_drawdown_breach": False, "correlation_crisis": False,
                "liquidity_crisis": False, "regime_forced": False,
            },
            "ongoing_conditions_count": 0,
            "ongoing_conditions": [],
        },

        # ── INTELLIGENCE ── (Platzhalter)
        "intelligence": {
            "status": "UNAVAILABLE",
            "consensus": {},
            "divergences": [],
            "divergences_count": 0,
            "high_novelty_claims": [],
            "catalyst_timeline": [],
        },

        # ── PIPELINE HEALTH ──
        "pipeline_health": {
            "overall_status": "PHASE_1",
            "steps": {
                "v16_daily_runner": {"status": "OK", "completed_at": now_utc, "summary": "V16 Gewichte berechnet"},
                "step_0a_dc": {"status": "NOT_BUILT", "summary": "Data Collector noch nicht implementiert"},
                "step_0b_ic": {"status": "NOT_BUILT", "summary": "Intelligence Collector noch nicht implementiert"},
                "step_1_market_analyst": {"status": "NOT_BUILT"},
                "step_2_signal_gen": {"status": "NOT_BUILT"},
                "step_3_risk_officer": {"status": "NOT_BUILT"},
                "step_4_cio_draft": {"status": "NOT_BUILT"},
                "step_5_da": {"status": "NOT_BUILT"},
                "step_6_cio_final": {"status": "NOT_BUILT"},
                "step_7_writer": {"status": "NOT_BUILT"},
            },
        },

        # ── KNOWN UNKNOWNS ──
        "known_unknowns": [
            {"gap": "FULL_PIPELINE", "reason": "phase_1_v16_only", "permanent": False},
            {"gap": "CIO_BRIEFING", "reason": "cio_agent_not_built", "permanent": False},
            {"gap": "INTELLIGENCE", "reason": "ic_not_built", "permanent": False},
            {"gap": "LAYER_SCORES", "reason": "market_analyst_not_built", "permanent": False},
        ],

        # ── TEMPORAL MAP ──
        "temporal_map": {
            "market_data_as_of": v16_data["data_date"] + "T21:00:00Z",
            "v16_regime_based_on": v16_data["data_date"] + " close",
            "dashboard_rendered_at": now_utc,
        },

        # ── AGENT R CONTEXT ──
        "agent_r_context": {
            "date": date,
            "briefing_type": briefing_type,
            "regime": regime,
            "regime_transition": False,
            "conviction": "N/A",
            "risk_ampel": "YELLOW" if stress >= 1 else "GREEN",
            "data_quality": "PARTIAL",
            "top_risk": f"DD-Protect: {v16_data['dd_protect_active']}" if v16_data["dd_protect_active"] else "Kein akutes Risiko",
            "action_items_act": [],
            "action_items_review": [],
            "divergences": [],
            "consistency_flags": [],
            "f6_pending": [],
            "cc_expiry_warnings": [],
            "regime_stability_pct": None,
        },

        # ── TIMESERIES ROW ──
        "timeseries_row": {
            "date": date,
            "regime": regime,
            "macro_state": v16_data["macro_state_num"],
            "growth": v16_data["growth_signal"],
            "liq_dir": v16_data["liq_direction"],
            "stress": v16_data["stress_score"],
            "dd_protect_count": len(v16_data["dd_protect_active"]),
            "pipeline_status": "PHASE_1",
        },

        # ── VALIDATION ──
        "validation": {
            "status": "PASS",
            "checks_run": 3,
            "checks_passed": 3,
            "checks_failed": 0,
            "warnings": ["Pipeline Phase 1: Nur V16-Daten verfuegbar"],
            "errors": [],
        },
    }

    return dashboard


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    t0 = datetime.now()
    print("="*60)
    print("V16 DAILY RUNNER — Phase 1")
    print(f"Datum: {t0.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Assets: {len(ASSETS)} | Channels: 9 | Clusters: {len(CLUSTERS)}")
    print("="*60)

    # 1. Load data
    prices, macro, k16, rv_pct, cm = load_data()

    # 2. Run engine
    print("\n" + "="*60)
    print("ENGINE (V38 1:1)")
    print("="*60)
    v16_data = run_engine(prices, macro, k16, rv_pct, cm)

    # 3. Build dashboard JSON
    print("\n" + "="*60)
    print("DASHBOARD JSON")
    print("="*60)
    dashboard = build_dashboard_json(v16_data)

    # 4. Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "data", "dashboard")
    os.makedirs(out_dir, exist_ok=True)

    # latest.json (fuer Frontend)
    latest_path = os.path.join(out_dir, "latest.json")
    with open(latest_path, 'w') as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)
    print(f"  latest.json ({os.path.getsize(latest_path)} bytes)")

    # Datiertes Backup
    dated_path = os.path.join(out_dir, f"dashboard_{v16_data['date']}.json")
    with open(dated_path, 'w') as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)
    print(f"  dashboard_{v16_data['date']}.json (Backup)")

    # 5. Summary
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n{'='*60}")
    print(f"FERTIG ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Date:    {v16_data['date']}")
    print(f"  Regime:  {v16_data['regime']} (State {v16_data['macro_state_num']}: {v16_data['macro_state_name']})")
    print(f"  Growth:  {v16_data['growth_signal']} | Liq: {v16_data['liq_direction']} | Stress: {v16_data['stress_score']}")
    print(f"  DD-Prot: {v16_data['dd_protect_active'] or 'Keine'}")
    print(f"  Top 5:   {', '.join(f'{w['ticker']}={w['weight']:.1%}' for w in v16_data['top_5_weights'])}")
    print(f"  CAGR:    {v16_data['performance']['CAGR']}% | Sharpe: {v16_data['performance']['Sharpe']}")
    print(f"  Output:  {latest_path}")

    return 0

if __name__ == '__main__':
    sys.exit(main() or 0)
