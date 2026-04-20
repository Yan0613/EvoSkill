---
name: livecodebench-solving
description: Solve LiveCodeBench competitive programming problems. Covers problem parsing, solution structure, edge case handling, and output format.
---

## Goal
Write a correct, efficient Python solution that passes all test cases.
Return the solution wrapped in `<FINAL_ANSWER>...</FINAL_ANSWER>` tags.

## Step-by-Step Workflow

1. **Parse the problem** — identify: input format, output format, constraints (n ≤ ?), time limit.
2. **Work through the examples** by hand to confirm your understanding.
3. **Choose an algorithm** — pick the simplest approach that fits within constraints:
   - n ≤ 1000 → O(n²) is fine
   - n ≤ 10⁵ → need O(n log n) or O(n)
   - n ≤ 10⁶ → need O(n) or O(n log n) with small constants
4. **Write the solution** — use standard Python I/O.
5. **Mentally trace** through all provided examples.
6. **Check edge cases**: empty input, n=1, all same values, maximum n.

## Solution Template

```python
import sys
input = sys.stdin.readline

def solve():
    # Read input
    n = int(input())
    # ... your logic ...
    print(answer)

t = int(input())
for _ in range(t):
    solve()
```

## Common Patterns

```python
# Read a line of integers
a = list(map(int, input().split()))

# Prefix sums
prefix = [0] * (n + 1)
for i in range(n):
    prefix[i+1] = prefix[i] + arr[i]

# Binary search
import bisect
idx = bisect.bisect_left(sorted_arr, target)

# BFS
from collections import deque
q = deque([start])
visited = {start}
while q:
    node = q.popleft()
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            q.append(neighbor)
```

## Output Format

- Match the expected output **exactly** — including case, spacing, and newlines.
- If the problem says "print YES or NO", do not print "yes" or "Yes".
- For multiple test cases, print each answer on a separate line.

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Off-by-one in loops/indices | Trace through example manually |
| Integer overflow (rare in Python) | Not an issue in Python |
| Slow I/O for large input | Use `sys.stdin.readline` |
| Wrong output format | Re-read output specification |
| Missing edge case (n=0, n=1) | Add explicit handling |
