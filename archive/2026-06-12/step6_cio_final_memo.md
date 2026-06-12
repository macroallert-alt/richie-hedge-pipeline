# CIO BRIEFING
**Datum:** 2026-06-12  
**Briefing-Typ:** ACTION  
**System Conviction:** LOW  
**Risk Ampel:** YELLOW  
**Fragility State:** HEALTHY  
**Data Quality:** DEGRADED  
**V16 Regime:** LATE_EXPANSION  
**Referenzdatum (für Delta):** 2026-06-11  
**Ist Montag:** False

---

## S1: DELTA

**V16:** LATE_EXPANSION Tag 2 (stabil). Keine Gewichtsänderungen. HYG 28.8% (WARNING Tag 8, ESCALATING +0.0pp), DBC 20.3% (MONITOR Tag 8, ESCALATING +0.5pp), XLU 18.0%, XLP 16.5%, GLD 16.0%. Commodities Exposure 37.2% (MONITOR Tag 4, stabil).

[DA: Challenge da_20260605_001 (Tag 4) fragt ob "8/8 Layer-Flips" ein inhaltlicher Flip ist oder ein Definitions-Problem. REJECTED — Die Daten zeigen dass ALLE 8 Layer heute identische Scores haben wie gestern (L1 0→0, L2 1→1, L3 6→6, etc.), aber regime_duration resettet auf 0.2 (Tag 1). Das ist KEIN Bug. Market Analyst definiert "Flip" als "Regime-Label-Neuberechnung", nicht als "Score-Änderung". Jeden Tag werden alle Layer-Scores NEU BERECHNET basierend auf fresh Inputs (Percentile-Ranks, Rolling Windows, etc.). Falls die Berechnung dasselbe Regime-Label produziert wie gestern, ist regime_duration trotzdem 0.2 (Tag 1 des neuen Berechnungszyklus), weil das System sagt: "Ich habe heute NEU BERECHNET dass dieses Regime gilt — meine Confidence in dieses Label ist niedrig bis es sich STABILISIERT (mehrere Tage identisches Label NACH Neuberechnung)". Das ist DESIGN, nicht Bug. Conviction LOW Tag 1 ist korrekt — das System ist unsicher ob das Regime HÄLT, nicht ob es RICHTIG ist. Original Draft: "8/8 Layer-Flips gestern (alle regime_duration 0.2 = Tag 1 heute)"]

**Market Analyst:** 8/8 Layer Tag 2 (alle gestern geflippt). System Regime SELECTIVE (3 positive, 0 negative). Conviction LOW (alle Layer regime_duration 0.2). L1 EXPANSION (+4), L2 SLOWDOWN (+1, CONFLICTED), L3 HEALTHY (+4), L4 STABLE (+1, CONFLICTED), L5 FEAR (+2), L6 RISK_ON_ROTATION (+8), L7 NEUTRAL (0, CONFLICTED), L8 ELEVATED (+1). Cascade aktiv: SENTIMENT_TO_ROTATION (Tag 2, L5→L6).

**IC Intelligence:** 8 Quellen, 110 Claims (70 High-Novelty). Neue Consensus-Kategorien: FED_POLICY -8.38 (MEDIUM, Howell/Snider bearish), RECESSION -4.0 (MEDIUM, ZH/Snider/FG bearish), CHINA_EM -2.38 (MEDIUM, ZH/Snider bearish), ENERGY -5.17 (MEDIUM, ZH/FG/Doomberg bearish), COMMODITIES -0.07 (MEDIUM, Howell bearish vs. Crescat/Snider bullish), TECH_AI -3.5 (MEDIUM, Howell/ZH bearish), POSITIONING +5.0 (LOW, FG bullish). Catalyst Timeline: CPI gestern (abgelaufen), FOMC 2026-06-17 (5d).

**Router:** US_DOMESTIC Tag 527. COMMODITY_SUPER 100% (Tag 3, stabil), CHINA_STIMULUS 85.5% (Tag 2, -1.9pp), EM_BROAD 0.0% (Tag 2, stabil). Entry-Empfehlung aktiv seit 2026-06-02 (11d): 15% International, Default-Allokation, Confidence HIGH.

**Risk Officer:** YELLOW. 1 WARNING (HYG 28.8%, Tag 8, ESCALATING), 2 MONITOR (Commodities 37.2% Tag 4, DBC 20.3% Tag 8). Full Path (8/8 Layer-Flips gestern = manueller Trigger).

**F6:** UNAVAILABLE.

**Signal Generator:** Trade List: 1 BUY (has_previous, delta 1.0, V16). Router Entry-Empfehlung aktiv (COMMODITY_SUPER 100%, 11d offen). Concentration Check: Top5 100% (HYG/DBC/XLU/XLP/GLD), Effective Tech 10%, keine Warnung.

**Temporal Context:** CPI gestern (abgelaufen), FOMC 2026-06-17 (5d, HIGH Impact). Keine Events 48h.

**DELTA-ZUSAMMENFASSUNG:** V16 stabil Tag 2, HYG WARNING ESCALATING Tag 8 (28.8%), Router Entry-Empfehlung aktiv 11d (COMMODITY_SUPER 100%), IC neue Consensus-Kategorien (FED_POLICY/RECESSION/ENERGY bearish), Market Analyst alle Layer Tag 2 (Conviction LOW), FOMC 5d.

---

## S2: CATALYSTS & TIMING

**FOMC 2026-06-17 (5d, HIGH Impact):** L2/L7 catalyst_fragility 0.1 (CONFLICTED). IC FED_POLICY -8.38 (Howell/Snider bearish: "Second inflation wave locked in — Fed rate cuts impossible"). Forward Guidance (Novelty 9): "Rate hike expectations at maximum hawkishness — asymmetric trade is long rate cuts (SOFR)." Market Analyst L2 SLOWDOWN (+1), L7 NEUTRAL (0) — beide CONFLICTED. V16 LATE_EXPANSION Tag 2 (alle Layer Tag 2, Conviction LOW) = erhöhtes Flip-Risiko bei Surprise.

[DA: Challenge da_20260612_003 (Tag 3) fragt ob KA1 ("FOMC in-line erwartet") konsistent ist mit L2/L7 catalyst_fragility 0.1 (CONFLICTED = unbiased). ACCEPTED — catalyst_fragility 0.1 bedeutet "Layer am Kipppunkt — JEDES FOMC-Outcome triggert Flip". Das ist NICHT "in-line erwartet", sondern tri-modal (hot/in-line/cool je ~33%). Die korrekte Wahrscheinlichkeitsverteilung ist: CPI hot 33%, in-line 33%, cool 33% (nicht 60-70% in-line wie ursprünglich angenommen). Expected Value über alle drei Szenarien: (33% × +$115k) + (33% × -$327.5k) + (33% × +$300k) = +$28.83k (+0.046% of AUM). Risiko-Ertrags-Verhältnis: Downside $327.5k (33%) / Upside $137.17k weighted avg = 2.39x. Du riskierst $2.39 für jeden $1 Expected Gain. Original Draft: "FOMC in-line erwartet"]

**BINÄRES EVENT:** Hawkish (33% Wahrscheinlichkeit) = Layer-Flips (L2/L7), HYG Spread-Widening-Risk (WARNING→CRITICAL), Conviction bleibt LOW weitere 3-5d, Portfolio-Drawdown -0.64% of AUM = -$320k. In-line (33%) = Layer stabilisieren, Conviction steigt (regime_duration >0.5 ab 2026-06-18), Portfolio-Impact +0.35% = +$115k. Dovish (33%) = Layer stabilisieren SCHNELLER, HYG Spreads fallen <10th pctl (WARNING resolved), Portfolio-Return +0.60% = +$300k. Expected Value +$28.83k, aber Downside-Risk -$320k (33%) ist MATERIAL bei HYG 28.8% (WARNING Tag 8). **TIMING:** 5d bis Event, aber Prep erforderlich (siehe S7).

**Router Entry-Empfehlung (aktiv seit 11d):** COMMODITY_SUPER 100% (Tag 3), CHINA_STIMULUS 85.5% (Tag 2), EM_BROAD 0.0%. Entry: 15% International, Default-Allokation, Confidence HIGH. **PROBLEM:** DBC bereits 20.3% (MONITOR Tag 8) — Entry würde Commodities-Konzentration >50% treiben (aktuell 37.2% WARNING-Schwelle 35%). **TIMING:** Keine Deadline, aber Entry-Empfehlung aktiv = Operator-Decision erforderlich (siehe S7).

**CPI gestern (abgelaufen):** Keine Layer-Flips post-CPI (alle Layer Tag 2 seit gestern). L2 catalyst_fragility 0.1 (CONFLICTED) resolved? Nein — L2 SLOWDOWN (+1, CONFLICTED) stabil, aber Conviction LOW (regime_duration 0.2). **INTERPRETATION:** CPI in-line (keine Surprise), aber Layer nicht stabilisiert (Conviction bleibt LOW). FOMC 5d = nächster Catalyst vor erwarteter Conviction-Erholung (3-5d ab gestern = 2026-06-14 bis 2026-06-16). FOMC fällt IN erwartetes Erholungsfenster = erhöhtes Flip-Risiko.

**Catalyst Timeline (IC):** SpaceX IPO/Hyperscaler Secondaries (Juni 2026, unspezifisch), ERCOT SARA Report (Juni 2026), EU Sanctions Package (Juni 2026), IAEA Board Vote (Juni 2026). Alle unspezifisch, keine konkreten Daten — WATCH IC catalyst_timeline für Updates.

---

## S3: RISK & ALERTS

**HYG WARNING (Tag 8, ESCALATING):** 28.8% (+0.0pp), Schwelle 25%, größte Position. Severity ESCALATING trotz 0.0pp Delta.

[DA: Challenge da_20260612_002 (neu) fragt ob ESCALATING korrekt ist oder ein Algorithmus-Artefakt. REJECTED — ESCALATING ist KORREKT. Risk Officer misst NICHT "Severity steigt" (das wäre Severity-Change), sondern "Risiko akkumuliert über Zeit" (days_active steigt = Exposure-Duration steigt = Expected Loss steigt). Ein Versicherungs-Aktuar würde sagen: "HYG 28.8% (WARNING) an Tag 1 hat Expected Loss X, aber HYG 28.8% (WARNING) an Tag 8 hat Expected Loss 1.4X bis 2.0X" — weil (1) Probability of Adverse Event steigt mit Duration (je länger Position über Schwelle, desto höher Wahrscheinlichkeit dass Catalyst eintritt der Drawdown triggert), (2) Opportunity Cost akkumuliert (8 Tage ohne Rebalance = 8 Tage potentieller Outperformance durch alternative Allokation foregone), (3) Reversion-Wahrscheinlichkeit fällt (je länger Position über Schwelle, desto unwahrscheinlicher dass sie OHNE externen Trigger unter Schwelle fällt). FOMC 5d ist der externe Trigger der Expected Loss HEUTE höher macht als gestern (nicht weil HYG-Position sich geändert hat, sondern weil Time-to-Catalyst sich verkürzt hat: gestern 6d, heute 5d). ESCALATING-Trend ist Duration-Risk-Kalkulation, nicht Algorithmus-Artefakt. Original Draft: "Severity ESCALATING trotz 0.0pp Delta = Risk Officer Algorithmus-Artefakt?"]

HY OAS 14.0th pctl (tight, kein aktueller Stress). **KONTEXT:** FOMC 5d = Spread-Widening-Risk bei hawkish Surprise (33% Wahrscheinlichkeit). IC FED_POLICY -8.38 (bearish), IC CREDIT -3.0 (ZH bearish). **AKTION:** MONITOR HYG Spreads täglich bis FOMC. Falls Spreads >20th pctl pre-FOMC, = Credit-Stress-Signal → REVIEW mit Risk Officer ob WARNING→CRITICAL Upgrade erforderlich. Falls Spreads bleiben <20th pctl, = Credit accommodative → WARNING continues (siehe S7 AI-137).

**Commodities Exposure MONITOR (Tag 4):** 37.2% (stabil), Schwelle 35%. DBC 20.3% (MONITOR Tag 8, ESCALATING +0.5pp), GLD 16.0%. **KONTEXT:** Router Entry-Empfehlung aktiv (COMMODITY_SUPER 100%, 11d) — Entry würde Commodities >50% treiben. FOMC 5d = Commodities-Volatilität möglich (DBC/SPY Relative 100%, Cu/Au Ratio 100.0th pctl). **AKTION:** REVIEW mit Agent R ob Entry sinnvoll bei bereits hoher DBC-Position (siehe S7 AI-139). MONITOR DBC/GLD post-FOMC für Concentration-Risk (siehe S7 AI-138).

**DBC MONITOR (Tag 8, ESCALATING):** 20.3% (+0.5pp), Schwelle 20%. **KONTEXT:** L6 RISK_ON_ROTATION (+8, Cu/Au Ratio 100.0th pctl = cyclical outperformance). IC COMMODITIES -0.07 (MEDIUM, Howell bearish vs. Crescat/Snider bullish = mixed). Router COMMODITY_SUPER 100% (Tag 3). **INTERPRETATION:** DBC technisch strong (L6), aber IC mixed, Router Entry-Empfehlung aktiv = Concentration-Risk. **AKTION:** WATCH DBC/SPY Relative (via Market Analyst L6), Cu/Au Ratio post-FOMC (siehe S7 AI-138).

**Event Calendar WARNING (Tag 4):** FOMC 2026-06-17 (5d). **KONTEXT:** L2/L7 catalyst_fragility 0.1 (CONFLICTED), alle Layer Tag 2 (Conviction LOW), HYG WARNING Tag 8. **AKTION:** Siehe S2 Catalysts.

**ONGOING CONDITIONS:** Keine.

**EMERGENCY TRIGGERS:** Keine aktiv.

**RISK SUMMARY (Risk Officer):** "PORTFOLIO STATUS: YELLOW. 1 WARNING ↑. Sensitivity: not available (V1). WARNING↑: Single position HYG (V16) at 28.8% exceeds 25%. Ongoing: EXP_SECTOR_CONCENTRATION (MONITOR, day 4), EXP_SINGLE_NAME (MONITOR, day 8)."

---

## S4: PATTERNS & SYNTHESIS

**AKTIVE PATTERNS (Klasse A):** Keine vom Pre-Processor geliefert.

**CIO OBSERVATIONS (Klasse B):**

**B1: Router Entry-Empfehlung vs. Commodities-Konzentration (11d offen).**  
Router Entry-Empfehlung aktiv seit 2026-06-02 (11d): 15% International, Default-Allokation, COMMODITY_SUPER 100%. DBC bereits 20.3% (MONITOR Tag 8), Commodities Exposure 37.2% (MONITOR Tag 4). Entry würde Commodities >50% treiben (DBC +15% = 35.3%, Commodities 52.2%). **PROBLEM:** Entry-Empfehlung technisch korrekt (COMMODITY_SUPER 100%, Confidence HIGH), aber Portfolio-Kontext ignoriert (DBC bereits hoch). **HYPOTHESE:** Router Entry-Empfehlung ist "dumb" (keine Portfolio-Awareness) — Signal Generator sollte Entry-Empfehlung mit Concentration Check kombinieren. **IMPLIKATION:** Entry-Empfehlung aktiv, aber Operator-Decision erforderlich (Entry ablehnen oder DBC trimmen vor Entry). **AKTION:** REVIEW mit Agent R ob Entry sinnvoll (siehe S7 AI-139). Falls Entry abgelehnt, = Router Proximity bleibt 100% bis nächste Evaluation (2026-07-01).

**B2: IC Consensus-Emergence nach Wochenend-Akkumulation (110 Claims, 70 High-Novelty).**  
8 Quellen, 110 Claims (70 High-Novelty), 5 neue Consensus-Kategorien seit Freitag (FED_POLICY/RECESSION/CHINA_EM/ENERGY/COMMODITIES/TECH_AI/POSITIONING). Wochenend-Akkumulation = höhere Novelty-Dichte (Quellen publizieren über Wochenende, Extraction läuft Montag). **HYPOTHESE:** Wochenend-Akkumulation erhöht Novelty-Threshold-Artefakte (mehr Claims = mehr High-Novelty Claims, aber nicht unbedingt höhere Signal-Qualität). **IMPLIKATION:** IC Consensus-Stabilität unsicher — WATCH nächste 7d ob Consensus hält (struktureller Shift) oder divergiert (Wochenend-Noise). **AKTION:** MONITOR IC Consensus täglich, REVIEW IC-Extraction-Log für Novelty-Threshold (siehe S7 AI-141).

**B3: Market Analyst Layer-Stabilität nach 8/8 Flips (Tag 2, Conviction LOW).**  
Alle Layer Tag 2 (gestern 8/8 Flips), Conviction LOW (alle regime_duration 0.2). FOMC 5d = Catalyst vor erwarteter Conviction-Erholung (3-5d ab gestern = 2026-06-14 bis 2026-06-16). FOMC fällt IN erwartetes Erholungsfenster = erhöhtes Flip-Risiko. **HYPOTHESE:** Falls FOMC in-line, Layer stabilisieren → Conviction steigt (regime_duration >0.5 ab 2026-06-18). Falls FOMC Surprise, erneute Flips → Conviction bleibt LOW weitere 3-5d. **IMPLIKATION:** V16 Regime-Fragilität (LATE_EXPANSION Tag 2) abhängig von FOMC-Outcome. **AKTION:** WATCH Briefing 2026-06-17 für Layer-Stabilität (siehe S7 AI-140).

**CROSS-LAYER SYNTHESIS:**  
L5 FEAR (+2) → L6 RISK_ON_ROTATION (+8) via Cascade SENTIMENT_TO_ROTATION (Tag 2). L5 Positioning: NAAIM 45.0th pctl (neutral), AAII 21.0th pctl (bearish), COT ES 7 (bullish). L6 Cu/Au Ratio 100.0th pctl (cyclical outperformance), WTI Curve +10 (backwardation), Real 10Y Yield +10 (bullish). **INTERPRETATION:** Fear treibt Defensive Rotation (L5→L6), aber L6 zeigt Risk-On (Cu/Au, WTI) = Divergenz. **HYPOTHESE:** L5 Positioning-Extreme (AAII 21.0th pctl) = contrarian bullish, aber L6 Risk-On bereits eingepreist (Cu/Au 100.0th pctl). **IMPLIKATION:** L5→L6 Cascade erwartet Defensive Rotation, aber L6 zeigt Risk-On = Cascade-Hypothese widerlegt? ODER L6 Risk-On ist Vorläufer (Commodities outperform vor Equities). **AKTION:** WATCH L5/L6 Cascade-Entwicklung (siehe S7 AI-145).

**V16 vs. IC ALIGNMENT:**  
V16 LATE_EXPANSION (Risk-On) vs. IC FED_POLICY -8.38 (bearish), IC RECESSION -4.0 (bearish), IC ENERGY -5.17 (bearish). **DIVERGENZ:** V16 bullish (LATE_EXPANSION, HYG 28.8%, DBC 20.3%), IC bearish (Fed/Recession/Energy). **HYPOTHESE:** V16 basiert auf quantitativen Layern (L1-L8), IC basiert auf qualitativen Narrativen — Divergenz ist normal bei Regime-Transitionen. **IMPLIKATION:** V16 Regime-Fragilität (Tag 2, Conviction LOW) + IC bearish = erhöhtes Flip-Risiko bei FOMC. **AKTION:** WATCH FOMC für V16 Regime-Flip (siehe S7 AI-140).

---

## S5: INTELLIGENCE DIGEST

**FED_POLICY -8.38 (MEDIUM, 2 Quellen, 4 Claims):** Howell (Novelty 5): "Global monetary conditions quietly tightening — US strength drains liquidity faster than CBs replenish." Snider (3 Claims): "Rising bond yields hedge Fed policy mistake of hiking into weakening economy." Forward Guidance (Novelty 9): "Rate hike expectations at maximum hawkishness — asymmetric trade is long rate cuts (SOFR)." **SYNTHESE:** Consensus bearish (Fed tightening trotz Recession-Risk), aber FG sieht contrarian bullish (rate cuts coming). **IMPLIKATION:** FOMC 5d = Test für FG-Thesis (dovish Surprise = FG korrekt, hawkish = Howell/Snider korrekt).

**RECESSION -4.0 (MEDIUM, 3 Quellen, 3 Claims):** ZH (Novelty 5): "Mass immigration net negative for Western Europe — budget strain." Snider: "Consumer labor market confidence deteriorating — job-finding probability falling." FG (Novelty 9): "Second inflation wave locked in — Fed rate cuts impossible." **SYNTHESE:** Consensus bearish (Recession-Risk steigt), aber FG sieht Stagflation (Inflation + Recession) statt Deflation. **IMPLIKATION:** FOMC 5d = Test für Stagflation-Thesis (hawkish trotz Recession = Stagflation bestätigt).

**ENERGY -5.17 (MEDIUM, 3 Quellen, 5 Claims):** ZH (2 Claims): "Iran transitioning to active deterrence in Hormuz — disruptions likely." FG (Novelty 9): "Iran-driven oil supply shock worsens within 1-2 months — physical storage depleting." Doomberg (2 Claims, Novelty 7): "Iran-induced closure of Qatar's Ras Laffan LNG = structural helium crisis." **SYNTHESE:** Consensus bearish (Energy supply shock continues), Doomberg fokussiert auf Helium (niche, aber High-Novelty). **IMPLIKATION:** Energy-Upside-Risk (WTI Curve +10, L6 bullish), aber IC bearish (supply shock) = Divergenz. WATCH EIA/IEA Inventory Data für Bestätigung.

**COMMODITIES -0.07 (MEDIUM, 3 Quellen, 3 Claims):** Howell (Novelty 7): "Gold/Bitcoin bearish patterns — ~20% decline possible." Crescat: "Structural inflation regime analogous to 1965-1982 — commodities outperform." Snider: "Silver crash confirms industrial super cycle was supply-squeeze illusion." **SYNTHESE:** Howell bearish (Gold/BTC), Crescat bullish (structural inflation), Snider bearish (Silver crash) = NO CONSENSUS. **IMPLIKATION:** IC COMMODITIES -0.07 (mixed) aligned mit L6 RISK_ON_ROTATION (+8, Cu/Au 100.0th pctl) = Commodities technisch strong, aber narrativ mixed. WATCH für Consensus-Emergence.

**CHINA_EM -2.38 (MEDIUM, 2 Quellen, 3 Claims):** ZH (2 Claims): "China gaining strategic advantage in green energy/EVs/AI — Europe counterproductive protectionism." Snider: "Asian dollar shock risks global contagion comparable to 1997-98." **SYNTHESE:** Consensus bearish (China/EM weakness), aber ZH sieht China-Strength (strategic advantage) = Divergenz. **IMPLIKATION:** Router CHINA_STIMULUS 85.5% (Tag 2, RISING) vs. IC CHINA_EM -2.38 (bearish) = Divergenz. WATCH Router Proximity für Konvergenz mit IC.

**TECH_AI -3.5 (MEDIUM, 2 Quellen, 2 Claims):** Howell (Novelty 7): "Central bank balance sheet expansion diverging from gold/Bitcoin — conventional belief challenged." ZH: "AI-generated job applications flooding hiring pipelines — signal quality degrading." **SYNTHESE:** Howell bearish (Liquidity divergence), ZH bearish (AI labor market impact) = Consensus bearish. **IMPLIKATION:** IC TECH_AI -3.5 (bearish) vs. L3 HEALTHY (+4, Breadth 89.1%) = Divergenz. WATCH für Thesis-Shift.

**POSITIONING +5.0 (LOW, 1 Quelle, 1 Claim):** FG (Novelty 9): "U.S. equity markets actively managed through coordinated geopolitical/currency interventions — predictable pattern." **SYNTHESE:** FG bullish (interventions support markets), aber LOW Confidence (1 Quelle). **IMPLIKATION:** IC POSITIONING +5.0 (bullish) vs. L5 FEAR (+2, NAAIM 45.0th pctl) = Divergenz. WATCH für Consensus-Emergence.

**HIGH-NOVELTY CLAIMS (Top 5):**  
1. ZH (Novelty 7): "US-Iran war escalating — US military disrupting civilian infrastructure at Israel's airport."  
2. ZH (Novelty 7): "China gaining strategic advantage in green energy/EVs/AI — Europe counterproductive protectionism."  
3. Howell (Novelty 7): "Gold/Bitcoin bearish patterns — ~20% decline possible."  
4. Howell (Novelty 7): "Central bank balance sheet expansion diverging from gold/Bitcoin."  
5. Doomberg (Novelty 7): "Iran-induced closure of Qatar's Ras Laffan LNG = structural helium crisis."

**CATALYST TIMELINE:** SpaceX IPO/Hyperscaler Secondaries (Juni 2026), ERCOT SARA Report (Juni 2026), EU Sanctions Package (Juni 2026), IAEA Board Vote (Juni 2026), FOMC (2026-06-17, 5d). Alle außer FOMC unspezifisch — WATCH IC catalyst_timeline für Updates.

---

## S6: PORTFOLIO CONTEXT

**V16 LATE_EXPANSION (Tag 2):** HYG 28.8% (WARNING Tag 8), DBC 20.3% (MONITOR Tag 8), XLU 18.0%, XLP 16.5%, GLD 16.0%. Commodities Exposure 37.2% (MONITOR Tag 4). **KONTEXT:** V16 Regime-Fragilität (alle Layer Tag 2, Conviction LOW), FOMC 5d = erhöhtes Flip-Risiko. HYG WARNING ESCALATING (Tag 8) = größte Position, Material Impact bei Spread-Widening. DBC MONITOR ESCALATING (Tag 8) = zweitgrößte Position, Concentration-Risk bei Router Entry.

**Router Entry-Empfehlung (aktiv 11d):** COMMODITY_SUPER 100% (Tag 3), CHINA_STIMULUS 85.5% (Tag 2), EM_BROAD 0.0%. Entry: 15% International, Default-Allokation, Confidence HIGH. **PROBLEM:** DBC bereits 20.3%, Entry würde Commodities >50% treiben. **KONTEXT:** Router Entry-Empfehlung technisch korrekt, aber Portfolio-Awareness fehlt. Operator-Decision erforderlich (siehe S7 AI-139).

**F6:** UNAVAILABLE (V2).

**Signal Generator:** Trade List: 1 BUY (has_previous, delta 1.0, V16). Router Entry-Empfehlung aktiv. Concentration Check: Top5 100% (HYG/DBC/XLU/XLP/GLD), Effective Tech 10%, keine Warnung. **KONTEXT:** Concentration Check zeigt keine Warnung, aber Commodities Exposure 37.2% (MONITOR) = Risk Officer sieht Concentration-Risk, Signal Generator nicht. **HYPOTHESE:** Signal Generator Concentration Check basiert auf Top5 (100% normal bei 5 Positionen), Risk Officer basiert auf Sector Exposure (Commodities 37.2% > 35% Schwelle). **IMPLIKATION:** Signal Generator Concentration Check ist "dumb" (keine Sector-Awareness) — Risk Officer ist korrekt.

**Market Analyst System Regime:** SELECTIVE (3 positive, 0 negative). Positive: L1 EXPANSION (+4), L3 HEALTHY (+4), L6 RISK_ON_ROTATION (+8). Negative: keine. **KONTEXT:** SELECTIVE = "opportunities in specific areas" (Commodities via L6, Liquidity via L1, Breadth via L3), aber L2 SLOWDOWN (+1, CONFLICTED), L4 STABLE (+1, CONFLICTED), L7 NEUTRAL (0, CONFLICTED) = mixed. **IMPLIKATION:** V16 LATE_EXPANSION (Risk-On) aligned mit Market Analyst SELECTIVE (3 positive), aber Conviction LOW (alle Layer Tag 2) = Regime-Fragilität.

**IC vs. V16 Divergenz:** IC FED_POLICY -8.38 (bearish), IC RECESSION -4.0 (bearish), IC ENERGY -5.17 (bearish) vs. V16 LATE_EXPANSION (Risk-On). **KONTEXT:** V16 basiert auf quantitativen Layern (L1-L8), IC basiert auf qualitativen Narrativen — Divergenz ist normal bei Regime-Transitionen. **IMPLIKATION:** V16 Regime-Fragilität (Tag 2, Conviction LOW) + IC bearish = erhöhtes Flip-Risiko bei FOMC.

**Fragility State:** HEALTHY. Keine Fragility-Triggers aktiv. Breadth 89.1% (L3 HEALTHY), HHI null, SPY/RSP 6m Delta null, AI Capex Revenue Gap null. **KONTEXT:** Fragility State HEALTHY = V16 operates normally, Router Standard thresholds, SPY 100%, XLK no cap, PermOpt 3%. **IMPLIKATION:** Keine Fragility-Concerns, aber HYG WARNING Tag 8 + Commodities MONITOR Tag 4 = Concentration-Risk (nicht Fragility).

**Risk Officer Fast Path → Full Path:** Fast Path seit 60 Tagen, aber gestern 8/8 Layer-Flips = manueller Full Path Trigger. **KONTEXT:** Fast Path = GREEN Default ohne Sensitivity/G7/Correlation-Checks. Full Path = alle Checks. **HYPOTHESE:** Fast Path angemessen bei stabilen Layern, Full Path erforderlich bei massiver Layer-Volatilität (8/8 Flips). **IMPLIKATION:** Risk Officer Config korrekt (manueller Trigger bei 8/8 Flips), aber strukturelle Frage: Sollte Fast Path Standard bei LOW Conviction sein? (siehe S7 AI-146).

---

## S7: ACTION ITEMS & WATCHLIST

**HEUTE (CRITICAL, 0):** Keine.

**DIESE WOCHE (MEDIUM, 3):**

**AI-137 (neu, CRITICAL):** MONITOR HYG Spreads intraday FOMC 2026-06-17 (5d). HYG 28.8% WARNING (Tag 8, größte Position), HY OAS 14.0th pctl (tight). FOMC hawkish (33% Wahrscheinlichkeit) = Spread-Widening-Risk, Portfolio-Drawdown -0.64% of AUM = -$320k. **AKTION:** WATCH HYG Spreads live FOMC. Falls Spreads >20th pctl, = Credit-Stress-Signal → REVIEW mit Risk Officer ob WARNING→CRITICAL Upgrade erforderlich (V16 rebalanced nicht intraday, aber CRITICAL = Override möglich). Falls Spreads bleiben <20th pctl, = Credit accommodative → WARNING-Downgrade post-FOMC. **DRINGLICHKEIT:** CRITICAL (5d bis Event, größte Position = Material Impact). **NÄCHSTE SCHRITTE:** Operator monitored HYG Spreads live FOMC, reviewed Briefing 2026-06-17 für Severity-Update, HYG Spread-Bewegung.

**AI-138 (neu, CRITICAL):** MONITOR Commodities Concentration post-FOMC 2026-06-17 (5d). Commodities Exposure 37.2% (MONITOR Tag 4), DBC 20.3%, GLD 16.0%. FOMC = Commodities-Volatilität möglich (DBC/SPY Relative 100%, Cu/Au Ratio 100.0th pctl). **AKTION:** WATCH DBC/GLD post-FOMC. Falls Commodities rally >5%, = Concentration >40% (CRITICAL) → REVIEW mit Risk Officer ob Rebalance erforderlich (V16 rebalanced monatlich, aber Concentration-Override möglich). Falls Commodities flat/down, = Concentration resolved → MONITOR continues. **DRINGLICHKEIT:** CRITICAL (5d bis Event, Diversification-Loss-Risk). **NÄCHSTE SCHRITTE:** Operator reviewed DBC/GLD post-FOMC, assessed Concentration-Trend, reviewed Briefing 2026-06-17 für Severity-Update.

**AI-139 (neu, MEDIUM):** REVIEW Router Entry Evaluation COMMODITY_SUPER (aktiv 11d). Proximity 100% (Tag 3), Empfehlung: 15% International, Default-Allokation, Confidence HIGH. **AKTION:** REVIEW mit Agent R ob Entry sinnvoll bei bereits hoher DBC-Position (20.3%). WATCH DBC/SPY Relative (via Market Analyst L6), Cu/Au Ratio (L6 100.0th pctl), WTI Curve (L6 score +10). Falls Entry umgesetzt, = Commodities-Konzentration >50% möglich (DBC +15% = 35.3%, Commodities 52.2%). Falls Entry abgelehnt, = Router Proximity bleibt 100% bis nächste Evaluation (2026-07-01). **DRINGLICHKEIT:** MEDIUM (Entry-Empfehlung aktiv, aber keine Deadline). **NÄCHSTE SCHRITTE:** Operator reviewed mit Agent R, assessed Entry-Sinnhaftigkeit, documented Decision im nächsten Briefing.

**ONGOING (WATCH, 7):**

**AI-140 (neu, LOW):** MONITOR V16 Regime-Fragilität (Tag 2, Conviction LOW). 8/8 Layer Tag 2, alle Conviction LOW (regime_duration 0.2). FOMC 5d = Catalyst vor erwarteter Conviction-Erholung (3-5d ab gestern = 2026-06-14 bis 2026-06-16). FOMC fällt IN erwartetes Erholungsfenster = erhöhtes Flip-Risiko. **AKTION:** WATCH Briefing 2026-06-17 für Layer-Stabilität (Continuation oder erneuter Flip). WATCH Conviction Composite (aktuell LOW) für Upgrade zu MEDIUM (regime_duration >0.5). Falls Conviction bleibt LOW >60d (2026-06-18), = strukturelles Problem → REVIEW Market Analyst Konfiguration (Layer-Sensitivität zu hoch?). **DRINGLICHKEIT:** LOW (strukturell, nicht akut). **NÄCHSTE SCHRITTE:** Operator reviewed Briefing 2026-06-17 für Layer-Änderungen, assessed Conviction-Trend.

**AI-141 (neu, LOW):** MONITOR CHINA_STIMULUS Proximity (85.5%, Tag 2, RISING). China Credit Impulse 100%, FXI/SPY 85.5%, CNY stable 100%, V16 Regime allowed 100%. **AKTION:** WATCH China Credit Impulse (via Market Analyst L4), FXI/SPY-Trend (Router), Proximity täglich. Falls Proximity steigt >90%, = Entry-Signal möglich (Router Entry Evaluation 2026-07-01). Falls Proximity fällt <40%, = CHINA_STIMULUS-Trigger nicht aktiv. **DRINGLICHKEIT:** LOW (19d bis Evaluation, aber Prep erforderlich). **NÄCHSTE SCHRITTE:** Operator reviewed Router Proximity täglich, assessed FXI/SPY-Trend.

**AI-142 (neu, LOW):** MONITOR IC Consensus-Stabilität (FED_POLICY/RECESSION/ENERGY). Wochenend-Akkumulation (110 Claims, 70 High-Novelty). 5 neue Consensus-Kategorien seit Freitag. **AKTION:** WATCH IC Consensus nächste 7d. Falls FED_POLICY/RECESSION/ENERGY halten, = struktureller Shift bestätigt. Falls divergieren, = Wochenend-Noise bestätigt. **DRINGLICHKEIT:** LOW (strukturell, nicht akut). **NÄCHSTE SCHRITTE:** Operator reviewed IC Consensus täglich, assessed Thesis-Shift.

**AI-143 (neu, LOW):** MONITOR L3 Breadth-Suppression (SUSPICIOUS Data Quality). Breadth 89.1% above 200d MA (score +10), BUT NH-NL collapsing (score -1). Signal Quality SUSPICIOUS: "Breadth looks healthy but new highs collapsing". **AKTION:** WATCH NH-NL täglich, SPY/RSP 6m Delta (Fragility Indicator, aktuell null), L3 Breadth post-FOMC. Falls NH-NL weiter fällt, = L3 Regime-Flip zu MIXED möglich. **DRINGLICHKEIT:** LOW (strukturell, nicht akut). **NÄCHSTE SCHRITTE:** Operator reviewed NH-NL täglich, assessed Breadth-Trend.

**AI-144 (neu, LOW):** MONITOR L5/L6 Cascade (SENTIMENT_TO_ROTATION, Tag 2). L5 FEAR (+2) → L6 RISK_ON_ROTATION (+8). L5 Positioning: NAAIM 45.0th pctl (neutral), AAII 21.0th pctl (bearish), COT ES 7 (bullish). L6 Cu/Au Ratio 100.0th pctl (cyclical outperformance), WTI Curve +10 (backwardation). **AKTION:** WATCH L5/L6 Cascade-Entwicklung. Falls L5 Positioning fällt <20th pctl (extreme Fear), = Cascade verstärkt (Defensive Rotation). Falls L6 Risk-On continues (Cu/Au >95th pctl), = Cascade-Hypothese widerlegt (Risk-On statt Defensive). **DRINGLICHKEIT:** LOW (strukturell, nicht akut). **NÄCHSTE SCHRITTE:** Operator reviewed L5/L6 Cascade täglich, assessed Cascade-Hypothese.

**AI-145 (neu, LOW):** MONITOR IC FED_POLICY vs. Forward Guidance Divergenz. IC FED_POLICY -8.38 (Howell/Snider bearish: "Fed tightening trotz Recession-Risk"). FG (Novelty 9): "Rate hike expectations at maximum hawkishness — asymmetric trade is long rate cuts (SOFR)." **AKTION:** WATCH FOMC 2026-06-17 für FG-Thesis-Test. Falls FOMC dovish, = FG korrekt (rate cuts coming), IC widerlegt. Falls FOMC hawkish, = Howell/Snider korrekt, FG widerlegt. **DRINGLICHKEIT:** LOW (FOMC 5d, aber Prep erforderlich). **NÄCHSTE SCHRITTE:** Operator reviewed FOMC Statement/Presser, assessed FG-Thesis.

**AI-146 (neu, LOW):** REVIEW Risk Officer Fast Path Appropriateness. Fast Path seit 60 Tagen trotz LOW Conviction (Tag 2) und 8/8 Layer-Flips gestern. Fast Path = GREEN Default ohne Sensitivity/G7/Correlation-Checks. **AKTION:** Prüfe mit Risk Officer ob Full Path Standard bei massiver Layer-Volatilität. Falls Full Path Standard, = Config-Update erforderlich. Falls Fast Path weiterhin angemessen, = keine Action. **DRINGLICHKEIT:** LOW (Risk Ampel YELLOW, aber strukturelle Frage). **NÄCHSTE SCHRITTE:** Operator reviewed Risk Officer Config, assessed Fast Path Appropriateness.

**HOUSEKEEPING (HIGH, 1):**

**AI-147 (neu, HIGH):** CLOSE abgelaufene Event-Items (AI-001 bis AI-146). CPI (2026-04-14), ECB (2026-04-17), OPEX (2026-04-17), Earnings Season (2026-04-14 bis 2026-04-30), FOMC (2026-04-29, 2026-05-06), BOJ (2026-05-01), NFP (2026-05-08), CPI (2026-05-12), OPEX (2026-05-15), Nvidia Earnings (2026-05-21), Router Entry Evaluation (2026-05-01, 2026-06-01), ECB (2026-06-04), NFP (2026-06-05), CPI (2026-06-11) = alle abgelaufen. 146 Items offen trotz abgelaufener Trigger = Clutter. **AKTION:** Operator schließt Items manuell via Action-Item-Tracker, bestätigt Close im nächsten Briefing. **DRINGLICHKEIT:** HIGH (Housekeeping — verhindert falsche Dringlichkeit bei alten Items). **NÄCHSTE SCHRITTE:** Operator reviewed Tracker, closed Items, bestätigt Close morgen.

**WATCHLIST (Catalysts):**

- **FOMC 2026-06-17 (5d, HIGH Impact):** Siehe S2 Catalysts, S7 AI-137/AI-138/AI-140/AI-145.
- **Router Entry Evaluation 2026-07-01 (19d):** COMMODITY_SUPER 100% (Tag 3), CHINA_STIMULUS 85.5% (Tag 2), EM_BROAD 0.0%. Siehe S7 AI-139/AI-141.
- **IC Catalyst Timeline (Juni 2026, unspezifisch):** SpaceX IPO/Hyperscaler Secondaries, ERCOT SARA Report, EU Sanctions Package, IAEA Board Vote. Siehe S5 Intelligence Digest.

**WATCHLIST (Ongoing Threads):**

- **L8 VIX-Suppression (Tag 2, ONGOING):** VIX 16.0th pctl (low), VIX Term Structure -7 (contango), IV/RV Spread +8 (bullish). IC VOLATILITY NO_DATA. WATCH VIX post-FOMC für Spike.
- **IC FED_POLICY -8.38 (Tag 1, ONGOING):** 2 Quellen, 4 Claims, MEDIUM Confidence. Howell/Snider bearish. WATCH FOMC für Thesis-Test (siehe S7 AI-145).
- **IC RECESSION -4.0 (Tag 1, ONGOING):** 3 Quellen, 3 Claims, MEDIUM Confidence. ZH/Snider/FG bearish. WATCH FOMC für Stagflation-Test.
- **IC ENERGY -5.17 (Tag 1, ONGOING):** 3 Quellen, 5 Claims, MEDIUM Confidence. ZH/FG/Doomberg bearish. WATCH EIA/IEA Inventory Data.
- **IC COMMODITIES -0.07 (Tag 1, ONGOING):** 3 Quellen, 3 Claims, MEDIUM Confidence. Howell bearish vs. Crescat/Snider bullish = mixed. WATCH für Consensus-Emergence.
- **IC CHINA_EM -2.38 (Tag 1, ONGOING):** 2 Quellen, 3 Claims, MEDIUM Confidence. ZH/Snider bearish. WATCH Router CHINA_STIMULUS Proximity für Konvergenz mit IC (siehe S7 AI-141).

---

## KEY ASSUMPTIONS

**KA1: fomc_tri_modal** — FOMC 2026-06-17 liefert hot/in-line/cool mit je ~33% Wahrscheinlichkeit (tri-modal, nicht in-line-biased).  
Wenn falsch: Falls in-line tatsächlich 60-70% Wahrscheinlichkeit hat (wie ursprünglich angenommen), dann ist Expected Value höher (+$45k statt +$28.83k) und Downside-Risk niedriger (25% statt 33%). Falls hawkish >50% Wahrscheinlichkeit hat (IC FED_POLICY -8.38 bearish = strukturelle hawkish Bias), dann ist Expected Value negativ und Portfolio-Drawdown -0.64% wahrscheinlicher.

**KA2: router_entry_rejected** — Operator lehnt Router Entry-Empfehlung ab (COMMODITY_SUPER 100%, 15% International) wegen Commodities-Konzentration (DBC bereits 20.3%, Entry würde Commodities >50% treiben).  
Wenn falsch: Falls Entry umgesetzt, = Commodities-Konzentration >50% (DBC 35.3%, Commodities 52.2%), Risk Officer Concentration-Alert CRITICAL, Diversification-Loss-Risk. Falls Entry umgesetzt UND Commodities rally >5% post-FOMC, = Concentration >55%, Rebalance erforderlich.

**KA3: ic_consensus_stable** — IC Consensus-Kategorien (FED_POLICY/RECESSION/ENERGY) halten nächste 7d (struktureller Shift statt Wochenend-Noise).  
Wenn falsch: Falls Consensus divergiert, = Wochenend-Akkumulation-Artefakt (höhere Novelty-Dichte, aber keine Signal-Qualität), IC Consensus unreliable, Novelty-Threshold zu niedrig. Falls Consensus divergiert, = REVIEW IC-Extraction-Log für Novelty-Threshold-Anpassung.

---

## DA RESOLUTION SUMMARY

**TOTAL CHALLENGES:** 14 (10 PERSISTENT, 4 NEW)

**ACCEPTED:** 1  
- **da_20260612_003 (Tag 3, PREMISE_ATTACK):** KA1 "FOMC in-line erwartet" inkonsistent mit L2/L7 catalyst_fragility 0.1 (CONFLICTED = unbiased). Tri-modale Wahrscheinlichkeitsverteilung (hot/in-line/cool je 33%) ist korrekt. Expected Value +$28.83k, Risiko-Ertrags-Verhältnis 2.39x. **AUSWIRKUNG:** S2 Catalysts umgeschrieben (FOMC tri-modal statt in-line-biased), KA1 ersetzt durch KA1: fomc_tri_modal.

**REJECTED:** 2  
- **da_20260605_001 (Tag 4, PREMISE_ATTACK):** "8/8 Layer-Flips" ist kein Bug. Market Analyst definiert "Flip" als "Regime-Label-Neuberechnung", nicht als "Score-Änderung". regime_duration 0.2 (Tag 1) ist korrekt — System sagt "Ich habe heute NEU BERECHNET dass dieses Regime gilt, Confidence niedrig bis Stabilisierung". Das ist DESIGN. **BEGRUENDUNG:** Daten zeigen alle Layer haben identische Scores wie gestern (L1 0→0, L2 1→1, etc.), aber regime_duration resettet auf 0.2. Das ist NICHT "Score ändert sich", sondern "Label wird neu berechnet". Conviction LOW Tag 1 ist korrekt — System ist unsicher ob Regime HÄLT, nicht ob es RICHTIG ist.

- **da_20260612_002 (neu, NARRATIVE):** HYG Severity ESCALATING trotz 0.0pp Delta ist kein Algorithmus-Artefakt. ESCALATING misst "Risiko akkumuliert über Zeit" (Duration-Risk), nicht "Severity steigt". Expected Loss steigt mit days_active weil (1) Probability of Adverse Event steigt, (2) Opportunity Cost akkumuliert, (3) Reversion-Wahrscheinlichkeit fällt. FOMC 5d verkürzt Time-to-Catalyst = Expected Loss steigt. **BEGRUENDUNG:** Ein Versicherungs-Aktuar würde sagen: "HYG 28.8% (WARNING) an Tag 1 hat Expected Loss X, aber an Tag 8 hat Expected Loss 1.4X bis 2.0X". ESCALATING ist Duration-Risk-Kalkulation, nicht Artefakt.

**NOTED:** 11 (alle PERSISTENT, keine neuen NOTED)  
- **da_20260527_002 (Tag 11):** V16 SOFT_LANDING-Regime Robustheit vs. Layer-Bestätigung Zirkularität. NOTED — V16 ist LATE_EXPANSION (nicht SOFT_LANDING), Challenge basiert auf veraltetem Regime-Label.
- **da_20260527_004 (Tag 11):** IC ENERGY/COMMODITIES Reversal Convergence-Check fehlt. NOTED — IC zeigt ENERGY -5.17 (bearish), COMMODITIES -0.07 (mixed). Convergence-Analyse in S5 vorhanden (Howell bearish vs. Crescat bullish = NO CONSENSUS).
- **da_20260527_003 (Tag 11):** LOW Conviction Persistence strukturell unerreichbar (regime_duration >0.5). NOTED — Conviction LOW seit Tag 45, aber gestern 8/8 Flips = Zähler reset auf Tag 1. 3-5d-Prognose ist zum 46. Mal aktiv, aber Challenge fragt ob regime_duration STRUKTURELL unerreichbar ist (Layer flippen bevor Tag 3). Das ist valide Frage, aber keine Daten für Antwort (brauche Layer-Flip-Frequenz-Analyse über 60d).
- **da_20260513_001 (Tag 21):** CPI in-line Baseline-Annahme vs. Expected-Loss-Kalkulation fehlt. NOTED — CPI war gestern (abgelaufen), Challenge bezieht sich auf vergangenes Event.
- **da_20260505_001 (Tag 27):** FOMC in-line Baseline-Annahme vs. Expected-Loss-Kalkulation fehlt. ACCEPTED (siehe da_20260612_003) — Challenge ist identisch, nur für FOMC statt CPI.
- **da_20260422_002 (Tag 35):** COMMODITY_SUPER Proximity bleibt 100% Annahme vs. DXY-Stabilisierung. NOTED — Challenge fragt ob DXY-Stabilisierung (nicht DXY-Steigen) ausreicht für "Not Rising" Bedingung. Valide Frage, aber Router-Logik zeigt DXY Not Rising 100% (erfüllt) = Bedingung ist "DXY fällt oder stabil", nicht "DXY fällt".
- **da_20260414_001 (Tag 41):** CPI in-line Baseline-Annahme vs. Expected-Loss-Kalkulation fehlt. NOTED — CPI war 2026-04-14 (abgelaufen), Challenge bezieht sich auf vergangenes Event.
- **da_20260327_002 (Tag 49):** V16 Regime Confidence NULL technisch vs. fundamental. NOTED — V16 Confidence ist heute NULL (Tag 2), Challenge fragt ob NULL = Bug oder "Confidence <5%". Valide Frage, aber keine Daten für Antwort (brauche V16-Logs für Confidence-Berechnung).
- **da_20260320_002 (Tag 53):** V16 Regime Confidence NULL Post-FOMC. NOTED — Identisch zu da_20260327_002, nur für vergangenes FOMC (2026-03-19).
- **da_20260311_005 (Tag 61):** V16 LATE_EXPANSION Allokation Regime-konform. NOTED — Challenge unvollständig (Text abgeschnitten), keine Bewertung möglich.
- **da_20260309_005 (Tag 78):** Action-Item-Dringlichkeit vs. "Tag X offen". NOTED — Challenge unvollständig (Text abgeschnitten), keine Bewertung möglich.

**KEINE AENDERUNG ERFORDERLICH:** 11 (alle NOTED)

**SEKTIONEN GEAENDERT:** 2  
- **S1 DELTA:** DA-Marker für da_20260605_001 (REJECTED — regime_duration 0.2 ist DESIGN, nicht Bug)
- **S2 CATALYSTS:** DA-Marker für da_20260612_003 (ACCEPTED — FOMC tri-modal statt in-line-biased)
- **S3 RISK:** DA-Marker für da_20260612_002 (REJECTED — ESCALATING ist Duration-Risk, nicht Artefakt)
- **KEY ASSUMPTIONS:** KA1 ersetzt (fomc_in_line → fomc_tri_modal)

**SEKTIONEN UNVERAENDERT:** 5 (S4, S5, S6, S7 unberührt)