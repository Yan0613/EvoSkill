---
name: officeqa-extraction
description: Extract numerical answers from U.S. Treasury Bulletin documents. Covers file selection, table parsing, fiscal-vs-calendar year handling, and answer formatting.
---

## Context

OfficeQA questions are about U.S. Treasury Bulletin data (expenditures, revenues, debt).
The task prompt lists absolute paths to local parsed `.txt` or `.json` files for the relevant bulletins.

## Step-by-Step Workflow

1. **Read the question** — identify the metric, the time period, and any qualifier ("calendar year", "fiscal year", "individual months").
2. **Identify which bulletin file(s) to read** from the paths in the task prompt.
3. **Extract the relevant table** — the bulletin files are large single-line JSON. Do NOT use Read or Grep directly on them (they will be blocked). Use `Bash` with `python3` to extract matching elements:
   ```
   python3 -c "import json; d=json.load(open('/path/to/file.json')); els=d['document']['elements']; [print(i, e['content'][:500]) for i,e in enumerate(els) if 'national defense' in str(e).lower()]"
   ```
   This prints only the relevant table rows, not the entire file.
4. **Extract the value** — be precise about row and column.
5. **Aggregate if needed** — if the question asks for a sum over months, sum each month's value.
6. **Return the number** in the format found in the source (usually with commas, in millions).

## Key Conventions

- **Fiscal year vs Calendar year**: U.S. fiscal year ends Sep 30. "Calendar year 1940" means Jan–Dec 1940.
- **"In millions of nominal dollars"**: return the number as-is (e.g. "2,602").
- **"Sum of individual calendar months"**: sum each monthly row — do **not** use the annual total row.
- **Multiple bulletins**: some questions span periods covered by multiple files — read all and combine.

## Reading Bulletin Files

The JSON bulletin files are large (300–900KB) and stored as a single line. **Never use Read or Grep directly** — they will be blocked. Always use `Bash` with `python3`:

```bash
# Find elements containing a keyword (returns only matching table snippets)
python3 -c "
import json
d = json.load(open('/path/to/bulletin.json'))
els = d['document']['elements']
for i, e in enumerate(els):
    if 'national defense' in str(e).lower():
        print(f'--- element {i} ---')
        print(e['content'][:800])
"

# Tables are HTML strings inside e['content']. Parse month rows like:
# <tr><td>1940-January</td><td>713</td><td>70</td><td>132</td>...</tr>
# Columns: Total, Departmental, National defense, Veterans' Admin, ...
```

## Answer Format

- Return only the number with commas (e.g. `2,602` or `44,463`).
- No units, no $ signs, no explanatory text.
- If summing, compute and return the total.

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Using annual total when asked for "sum of individual months" | Sum each monthly row manually |
| Confusing fiscal year with calendar year | Check the question wording carefully |
| Reading the wrong bulletin file | Match the question's time period to the bulletin's date |
| Returning "$2,602 million" | Strip to just "2,602" |
