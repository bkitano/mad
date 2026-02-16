# Research Log Tab - Added

## What Was Added

Added `notes/log.md` (Research Activity Log) as a 4th tab in the dashboard.

## Changes

### Backend
**Modified**: `dashboard/sse_server.py`
- Added `NOTES_DIR` and `RESEARCH_LOG` configuration
- Added `/api/research-log` endpoint

### Frontend
**New**:
- `blog/src/components/mad/ResearchLog.tsx` - Research log viewer

**Modified**:
- `blog/src/pages/MADDashboard.tsx` - Added "Research Log" tab

## What It Shows

The research log contains:
- **High-Impact Proposals** - MAD system's analysis of which proposals are most valuable
- **Experiment Updates** - Status and findings from completed experiments
- **Strategic Insights** - Priority rankings and recommendations
- **New Discoveries** - Tricks and techniques learned
- **Automated Analysis** - Impact scores, cost estimates, and reasoning

It's like the MAD system's journal - showing its thought process and decision-making.

## Features

- Auto-refreshes every 30 seconds
- Full markdown rendering with math equations
- Shows last updated timestamp
- Informative banner explaining what it is

## Usage

1. Click "Research Log" tab
2. Read the automated analysis and insights
3. See which proposals the system thinks are high-priority
4. View experiment results and strategic recommendations

## Example Content

From the log you'll see entries like:

```markdown
### 🎯 High-Impact Proposals

- **Monarch-Gated State Transition SSM** (Proposal 006 — Priority: **HIGH**)
  - **Impact score**: **9/10**
  - **Why it matters**: Fills the exact gap between diagonal SSMs...
  - **Estimated cost**: **<$3**

### 🧪 Experiment Updates

- **Experiment 002: Oscillatory-DPLR SSM** (Status: **completed — FAILED**)
  - Training MSE: 0.854 (target: <1e-3)
  - Verdict: DEBUG - gradient flow bug

### Strategic Insights

Revised priority order (optimizing for information value per dollar):
1. Monarch-Gated SSM (006) — <$3, highest novelty
2. Oscillatory-DPLR debug (004) — <$1, diagnose failure
...
```

## Dashboard Now Has 4 Tabs

1. **Experiments** - Real-time running experiments
2. **Proposals** - Browse 42 proposals with filters
3. **Tricks** - Search 200+ computational tricks
4. **Research Log** - MAD system's analysis and insights ✨ NEW

## Testing

```bash
# Test endpoint
curl http://localhost:8001/api/research-log | jq

# View in dashboard
npm run dev
open http://localhost:5173/mad
# Click "Research Log" tab
```

## Why This Is Valuable

The research log shows the MAD system's **reasoning** and **priorities**:
- Which proposals it thinks are most valuable
- Why certain experiments failed or succeeded
- Strategic recommendations for what to do next
- Cost/benefit analysis

It's like having a research assistant that documents its thinking!

---

**Total tabs**: 4
**New files**: 1 (ResearchLog.tsx)
**Modified files**: 2 (sse_server.py, MADDashboard.tsx)
