---
name: dabstep-analysis
description: Solve DABStep payment data analysis tasks using pandas over CSV/JSON files. Covers file navigation, business rule lookup, aggregation patterns, and strict answer formatting.
---

## Available Files

The task prompt lists absolute paths — use them directly:

| File | Purpose |
|------|---------|
| `manual.md` | Business rules (read first for rule-based questions) |
| `payments-readme.md` | Column definitions for payments.csv |
| `payments.csv` | Transaction-level data (large — use Bash+pandas, never Read) |
| `fees.json` | Fee schedule per MCC / acquirer / card scheme |
| `merchant_data.json` | Merchant metadata |
| `merchant_category_codes.csv` | MCC code → category mapping |
| `acquirer_countries.csv` | Acquirer → country mapping |

## Step-by-Step Workflow

1. **Read the question and guidelines** — the guidelines define the exact answer format.
2. **Read `manual.md`** when the question involves business rules, fee logic, or terminology.
3. **Use Bash+pandas** to analyze CSV/JSON files — never Read large CSV files directly.
4. **Aggregate / filter** to answer the question.
5. **Format the answer** exactly as the guidelines specify.

## Bash+Pandas Patterns

```bash
python3 << 'EOF'
import pandas as pd, json

df = pd.read_csv('/absolute/path/payments.csv')

# Highest transaction count by issuing country
print(df['issuing_country'].value_counts().idxmax())

# Top fraud country
fraud_df = df[df['has_fraud'] == True]
print(fraud_df['ip_country'].value_counts().idxmax())

# Merge MCC codes
mcc = pd.read_csv('/absolute/path/merchant_category_codes.csv')
merged = df.merge(mcc, on='mcc', how='left')
EOF
```

## Answer Formatting Rules

- **Country code only**: return the 2-letter ISO code (e.g. "NL"), not a full name.
- **Multiple choice**: return exactly "X. Y" where X is the letter and Y is the value (e.g. "B. BE").
- **Numeric**: return the number only, no units unless explicitly requested.
- **Not Applicable**: if the data genuinely does not contain an answer, return exactly "Not Applicable".

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Reading payments.csv with Read tool | Always use Bash+pandas |
| Guessing column names | Check `payments-readme.md` or print `df.columns` first |
| Wrong aggregation (sum vs count) | Re-read the question and guidelines |
| Returning "Netherlands" instead of "NL" | Check format the guidelines require |
