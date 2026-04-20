---
name: gdpval-deliverable
description: Create deliverable files (Excel, PowerPoint, CSV, Markdown) for GDPVal benchmark tasks. Covers rubric reading, file creation, and manifest output.
---

## Goal

GDPVal tasks require creating **output files**, not just text answers.
Read the reference files and rubric, create the deliverable in the `submission/` directory,
then return a file manifest inside `<FINAL_ANSWER>...</FINAL_ANSWER>` tags.

## Step-by-Step Workflow

1. **Read the task prompt** — note the expected file types and the `submission/` directory path.
2. **Read the reference files** listed in the prompt using the Read tool.
3. **Read the rubric** carefully — it defines what the deliverable must contain.
4. **Create the file(s)** in the `submission/` directory using Bash or Write.
5. **Return the manifest** — file names only, one per line, in `<FINAL_ANSWER>` tags.

## Creating Excel Files (.xlsx)

```bash
python3 << 'EOF'
import pandas as pd
df = pd.DataFrame({"Column A": [1, 2], "Column B": [3, 4]})
df.to_excel("/path/to/submission/output.xlsx", index=False)
EOF
```

## Creating CSV Files (.csv)

```bash
python3 << 'EOF'
import pandas as pd
df.to_csv("/path/to/submission/output.csv", index=False)
EOF
```

## Creating PowerPoint Files (.pptx)

```bash
python3 << 'EOF'
from pptx import Presentation
prs = Presentation()
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "Title"
prs.save("/path/to/submission/output.pptx")
EOF
```

## Creating Markdown Files (.md)

Use the Write tool directly for Markdown.

## Manifest Format (FINAL_ANSWER)

```
<FINAL_ANSWER>
output.xlsx
summary.md
</FINAL_ANSWER>
```

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Saving files outside `submission/` | Always use the exact submission path from the task |
| Skipping rubric criteria | Read every rubric item before creating files |
| Including explanation in FINAL_ANSWER | Return only file names, one per line |
