---
name: quantitative-analysis-methodology
description: >
  Comprehensive methodology guidance for quantitative financial analysis. MUST be triggered
  BEFORE: (1) risk calculations (ES, VaR, volatility), (2) forecasting (exponential smoothing,
  moving averages), (3) currency conversion, (4) federal budget classification. Enforces
  MANDATORY validation checkpoints with PASS/FAIL output before computation. Prevents
  systematic errors from wrong data transformations, date misalignment, notation parsing,
  or classification mistakes.
---

# Quantitative Analysis Methodology Guide

---

## ⚠️ MANDATORY FIRST OUTPUT — READ THIS BEFORE ANYTHING ELSE

**You MUST output a validation block as your VERY FIRST action, BEFORE extracting any data or performing any calculation.**

This is not optional. Skipping this step WILL result in wrong answers.

### Required First Output Format

**Immediately upon starting ANY quantitative task, output this block:**

```
═══════════════════════════════════════════════════════════
METHODOLOGY VERIFICATION - [Task Type: ES/VaR/Forecasting/Currency]
═══════════════════════════════════════════════════════════

Check 1 - Data Form (for ES/VaR tasks):
  Data form: Returns required.
  Will compute returns from levels before calculation: [YES/NO]
  Status: [PASS if YES / BLOCKED if NO]

Check 2 - Date Alignment (for currency conversion tasks):
  Price date: [exact date from source]
  Exchange rate date: [exact date from rate source]
  Match: [PASS if identical / BLOCKED if different]

VERIFICATION RESULT: [PROCEED / BLOCKED - must resolve before continuing]
═══════════════════════════════════════════════════════════
```

**You MUST output this validation block before any calculation. Skipping this step will result in wrong answers.**

---

## ⚠️ BLOCKING CONDITIONS — DO NOT PROCEED IF THESE APPLY

### BLOCKED: ES/VaR on Levels

**If you have yield/price LEVELS and will calculate ES/VaR directly on those levels, STOP.**

You MUST compute returns first. ES on levels is ALWAYS WRONG.

- ❌ WRONG: Extract yields [9.56%, 8.36%, 7.37%, 6.14%], sort them, return min (6.14%) as ES
- ✓ CORRECT: Extract yields, compute returns between periods, then calculate ES on returns

### BLOCKED: Date Misalignment

**If price date is "end of March" and exchange rate date is "March 1", STOP.**

These are NOT the same date. Find end-of-March exchange rates or document why they're unavailable.

- ❌ WRONG: Use March 1 exchange rate for end-of-March Treasury price
- ✓ CORRECT: Use end-of-March exchange rate for end-of-March Treasury price

---

## ⚠️ MOST COMMON ERRORS — CHECK BEFORE PROCEEDING

| Error # | Wrong Approach | Why It's Wrong | Correct Approach |
|---------|---------------|----------------|------------------|
| 1 | Taking min(yield levels) as ES | ES requires RETURNS, not levels | Compute returns first, then ES on returns |
| 2 | Using March 1 exchange rates for end-of-March prices | Dates must match exactly | Use end-of-March exchange rates |

**If your approach matches either error pattern above, STOP and reconsider.**

---

## Validation Checkpoint Templates

**Output the appropriate checkpoint BEFORE any multi-step financial calculation:**

### For ES/VaR/Volatility Calculations

```
VALIDATION CHECKPOINT - Risk Calculation
Check 1: Data Form
  - Required form: returns
  - Current form: [levels/returns]
  - Status: [PASS if returns / BLOCKED if levels - must compute returns first]
Check 2: Return Calculation (if transformed)
  - Formula used: [e.g., (V_t - V_{t-1}) / V_{t-1}]
  - Sample calculation: [show one example]
  - Status: [PASS if verified / BLOCKED if not shown]
Check 3: Output Sign Convention
  - Loss representation: negative returns
  - Status: [PASS if convention documented]
OVERALL: [PASS - proceed / BLOCKED - halt and resolve]
```

### For Currency Conversion

```
VALIDATION CHECKPOINT - Currency Conversion
Check 1: Date Alignment
  - Price observation date: [exact date from source]
  - Exchange rate date: [exact date from rate source]
  - Status: [PASS if identical / BLOCKED if different]
Check 2: Rate Direction
  - Quote convention: [e.g., "DEM per USD" or "USD per DEM"]
  - Operation to apply: [multiply/divide]
  - Status: [PASS if verified / BLOCKED if uncertain]
Check 3: Source Documentation
  - Price source: [document/table/cell reference]
  - Rate source: [document/API/reference]
  - Status: [PASS if documented / BLOCKED if assumed]
OVERALL: [PASS - proceed / BLOCKED - halt and resolve]
```

### For Forecasting

```
VALIDATION CHECKPOINT - Forecasting
Check 1: Notation Parsing
  - Raw value: [as appears in source]
  - Parsed value: [decimal equivalent]
  - Parsing method: [e.g., "32nds: 76 + 18/32 = 76.5625"]
  - Status: [PASS if shown / BLOCKED if assumed]
Check 2: Initialization Method
  - Method: [e.g., "F_1 = Y_1 (first observation)"]
  - Initial value: [numeric]
  - Status: [PASS if explicit]
Check 3: Error Convention
  - Formula: Error = Actual - Forecast
  - Status: [PASS if using standard convention]
OVERALL: [PASS - proceed / BLOCKED - halt and resolve]
```

---

## Part 1: Risk Metric Methodology

### Pre-Flight Checklist

Before ANY quantitative financial analysis:

1. **Identify the metric type** - Determines required data form
2. **Check data form** - Is it levels or returns? Prices or yields?
3. **Apply required transformation** - Convert if needed
4. **Validate output sign/magnitude** - Sanity check results

### Expected Shortfall (ES / CVaR)

**CRITICAL: ES is calculated on RETURNS, not raw values.**

| Data Form | Required Action | Example |
|-----------|-----------------|---------|
| Yield levels (9.56%, 8.36%, ...) | Compute returns FIRST | r_t = (Y_t - Y_{t-1}) / Y_{t-1} |
| Price levels ($100, $98, ...) | Compute returns FIRST | r_t = (P_t - P_{t-1}) / P_{t-1} |
| Return series (-2.5%, 1.2%, ...) | Use directly | No transformation |

**Formula:**
```
ES_alpha = E[Loss | Loss > VaR_alpha]
```

At alpha = 5%, ES is the average of the worst 5% of returns.

**Correct workflow for yield data:**
1. Extract yield levels: [9.56%, 9.60%, 8.36%, 7.37%, 7.68%, 6.69%, 5.87%, 6.04%, 5.65%, 6.14%]
2. Compute period-over-period returns: [(9.60-9.56)/9.56, (8.36-9.60)/9.60, ...]
3. Sort returns, take worst alpha percentile
4. Average those worst returns → ES

**Common error:** Taking min/max of raw levels. This is NOT ES.

### Value at Risk (VaR)

Same transformation rules as ES. VaR is the threshold loss at confidence level alpha.

```
VaR_alpha = quantile(returns, alpha)
```

At 95% confidence, VaR is the 5th percentile of returns.

---

## Part 2: Forecasting Methodology

### Bond Price Notation Parsing

**CRITICAL: Treasury bond prices are quoted in 32nds, not decimals.**

| Notation | Meaning | Decimal Value |
|----------|---------|---------------|
| 76.18 | 76 + 18/32 | 76.5625 |
| 99.16 | 99 + 16/32 | 99.5000 |
| 100.08 | 100 + 8/32 | 100.2500 |
| 98.24+ | 98 + 24.5/32 | 98.765625 |

**Conversion formula:**
```
Decimal Price = Integer Part + (Fractional Part / 32)
```

**The "+" suffix:**
- 98.24+ means 98 + 24.5/32 (adds 0.5 to the numerator)
- Equivalent to adding 1/64 to the base price

**Validation checks:**
- Fractional part must be 00-31 (values >=32 are invalid)
- Treasury note prices typically range 90-110
- Treasury bond prices can range more widely

**Common error:** Treating 76.18 as 76.18 decimal instead of 76 + 18/32 = 76.5625.

### Exponential Smoothing

**Formula:**
```
F_{t+1} = alpha * Y_t + (1-alpha) * F_t
```

Where:
- F_{t+1} = Forecast for period t+1 (made at end of period t)
- Y_t = Actual value observed in period t
- F_t = Forecast that was made for period t
- alpha = Smoothing parameter (0 < alpha <= 1)

**Initialization Methods:**

| Method | When to Use | Formula |
|--------|-------------|---------|
| First observation | Short series (<20 obs), simple | F_1 = Y_1 |
| Average of first k | Longer series, more stable | F_1 = mean(Y_1...Y_k) |
| Backcast | Optimal but complex | Requires iterative fitting |

**Default:** Use first observation method (F_1 = Y_1) unless otherwise specified.

**Forecast Timing Convention:**

**CRITICAL: A forecast for period t is made at the END of period t-1.**

### Forecast Error Conventions

**CRITICAL: Error = Actual - Forecast (NOT Forecast - Actual)**

```
e_t = Y_t - F_t
```

| Error Sign | Interpretation |
|------------|----------------|
| Positive (+) | Under-forecast (actual exceeded forecast) |
| Negative (-) | Over-forecast (actual was below forecast) |

**Example:**
- Actual (Y_t) = 190.73
- Forecast (F_t) = 210.02
- Error = 190.73 - 210.02 = **-19.29** (over-forecast)

**WRONG:** e_t = F_t - Y_t (reverses sign interpretation)

---

## Part 3: Currency Conversion

### Date Alignment Rule

**CRITICAL: Use the exchange rate from the SAME DATE as the price observation.**

| Price Date | Exchange Rate Date | Correct? |
|------------|-------------------|----------|
| End of March | End of March | Yes |
| End of March | March 1 | No |
| End of March | Beginning of April | No |

**End-of-Month Convention:**

For end-of-month price data:
- Use end-of-month exchange rates
- "End of month" typically means last business day
- Not "first of next month" or "average for month"

### Exchange Rate Direction

**Verify quote convention before applying:**

| Convention | Example | To convert USD to DEM |
|------------|---------|---------------------|
| DEM per USD | 1.85 DEM/USD | Multiply: USD * 1.85 |
| USD per DEM | 0.54 USD/DEM | Divide: USD / 0.54 |

**CRITICAL Date Alignment Rules:**
- "End of March" prices require "end of March" exchange rates
- "March 1" != "end of March" → BLOCKED
- "April 1" != "end of March" → BLOCKED
- Treasury Bulletin April edition contains END OF MARCH data

---

## Part 4: Government Accounting Classifications

**CRITICAL: Use OMB/Treasury definitions, not intuitive interpretation.**

For federal obligation classifications, see `references/federal-accounting.md`.

Key principle: "Service-related" in federal accounting has specific regulatory meaning per OMB Circular A-11 that differs from casual interpretation.

---

## Part 5: Data Transformation Rules

### When to Compute Returns

| Analysis Type | Requires Returns? | Why |
|---------------|-------------------|-----|
| ES / CVaR | YES | Risk metrics measure change, not level |
| VaR | YES | Same as ES |
| Volatility / Std Dev | Usually YES | Volatility of returns, not levels |
| Performance analysis | YES | Compare % gains/losses |
| Trend analysis | NO | Levels show trend direction |
| Point-in-time snapshot | NO | Current state, not change |

### Return Calculation Methods

**Simple returns** (default for most financial analysis):
```
r_t = (V_t - V_{t-1}) / V_{t-1}
```

**Log returns** (for compounding analysis):
```
r_t = ln(V_t / V_{t-1})
```

Use simple returns unless log returns are specifically required.

---

## Part 6: Multi-Source Data Combination

**Before combining time series from different sources:**

```
VALIDATION CHECKPOINT - Data Combination
Check 1: Frequency Match
  - Source A frequency: [monthly/quarterly/etc.]
  - Source B frequency: [monthly/quarterly/etc.]
  - Status: [PASS if identical / BLOCKED if different]
Check 2: Date Convention Match
  - Source A dates: [e.g., "end of month"]
  - Source B dates: [e.g., "end of month"]
  - Status: [PASS if identical / BLOCKED if different]
Check 3: Period Alignment
  - Sample period: [e.g., "March 1972"]
  - Source A date: [exact date]
  - Source B date: [exact date]
  - Status: [PASS if aligned / BLOCKED if misaligned]
OVERALL: [PASS - proceed / BLOCKED - halt and resolve]
```

---

## Part 7: Output Verification — BEFORE REPORTING RESULTS

### ES/VaR Output Check

**If your ES value equals the minimum of your raw data values, you made the levels-vs-returns error. Go back and compute returns.**

Example self-check:
- Raw yield values: [9.56, 9.60, 8.36, 7.37, 7.68, 6.69, 5.87, 6.04, 5.65, 6.14]
- Minimum raw value: 5.65
- Your ES answer: 5.65 ← **WRONG** — this equals min(raw), indicating ES was computed on levels
- Correct: Compute returns first, then ES on returns (will be a negative percentage)

### Forecasting Output Check

**If your forecast error is in the range of ±5 for DEM-converted bond prices, verify you used end-of-month exchange rates, not month-start rates.**

Systematic ±5 error often indicates exchange rate date misalignment when converting USD Treasury prices to DEM.

### Quick Verification Table

| Metric | Expected Characteristics | Red Flag |
|--------|-------------------------|----------|
| ES on returns | Negative value (for loss measurement) | Positive or equals raw data min |
| VaR on returns | Negative value at alpha < 50% | Positive or equals raw percentile of levels |
| Parsed 32nds price | Close to quoted integer | >1 point difference |
| Exponential smooth | Between min and max of data | Outside data range |
| Forecast error sign | Negative if over-forecast | Sign seems reversed |
| Currency conversion | Same order of magnitude | 10x or 0.1x original |
| Date alignment | Exact date match | Off by days |

---

## Quick Decision Tree

```
Starting a financial analysis task?
│
├── OUTPUT MANDATORY VERIFICATION BLOCK FIRST
│   └── If any check is BLOCKED → resolve before proceeding
│
├── Calculating ES, VaR, or volatility?
│   ├── Is data already in return form?
│   │   ├── YES → Proceed with calculation
│   │   └── NO → Compute returns first, then calculate
│   └── OUTPUT CHECK: Does ES equal min(raw_values)?
│       └── YES → WRONG — go back and compute returns
│
├── Doing forecasting?
│   ├── Is price data in 32nds notation?
│   │   ├── YES → Parse: Integer + Fractional/32
│   │   └── NO → Use as-is (verify it's decimal)
│   ├── Using exponential smoothing?
│   │   └── Initialize: F_1 = Y_1 (first observation)
│   └── Computing error?
│       └── Error = Actual - Forecast (signed)
│
├── Converting currencies?
│   ├── Match exchange rate date to price date exactly
│   │   └── "March 1" ≠ "end of March" → BLOCKED
│   ├── Verify rate direction (per USD or per foreign)
│   └── Validate converted magnitude is reasonable
│
└── Classification task?
    └── Check references/federal-accounting.md for definitions
```
