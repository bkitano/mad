# Proposal Status Workflow

This document describes the automatic status tracking for proposals as they move through the experiment pipeline.

## Status Values

### Manual Statuses (set by researcher)

- **`proposed`** - Initial status for new proposals. Default state.
- **`abandoned`** - Proposal deemed not worth pursuing. Set manually when deciding to skip.
- **`validated`** - Experiment succeeded and approach is proven. Set manually after reviewing results.
- **`needs-revision`** - Experiment showed promise but needs changes. Set manually based on results.

### Automatic Statuses (set by experiment agent)

- **`ongoing`** - Experiment implementation started. Set automatically when experiment agent begins work.
- **`completed`** - Experiment finished successfully. Set automatically when experiment agent completes without errors.
- **`failed`** - Experiment implementation or execution failed. Set automatically when experiment agent encounters errors.
- **`on-hold`** - Proposal temporarily skipped. Set automatically when experiment already exists or resource constraints prevent starting.

## Workflow

```
proposed
   │
   ├─→ ongoing (experiment agent starts)
   │     │
   │     ├─→ completed (experiment finishes successfully)
   │     │     │
   │     │     ├─→ validated (manual: results confirm hypothesis)
   │     │     └─→ needs-revision (manual: results show promise but need changes)
   │     │
   │     └─→ failed (experiment errors out)
   │
   ├─→ on-hold (automatic: experiment exists or resource constraints)
   │
   └─→ abandoned (manual: decision not to pursue)
```

## Usage

### Viewing Status in Dashboard

The MAD dashboard automatically shows status badges on each proposal card:
- Blue badge = `proposed`
- Yellow badge = `ongoing` or `on-hold`
- Green badge = `completed` or `validated`
- Red badge = `failed` or `abandoned`
- Gray badge = other statuses

Filter proposals by status using the dropdown in the proposals view.

### Manually Updating Status

Edit the YAML frontmatter at the top of any proposal file:

```yaml
---
status: proposed
priority: high
created: 2026-02-15
---
```

Change `status:` to any of the values above. The dashboard will reflect changes in real-time.

### Automatic Updates

The experiment agent automatically updates status:

1. **When starting implementation**: `proposed` → `ongoing`
2. **When successfully completing**: `ongoing` → `completed`
3. **When encountering errors**: `ongoing` → `failed`
4. **When skipping (already exists)**: `proposed` → `on-hold`

After automatic status updates, you should review results and manually update to:
- `validated` if experiment confirms hypothesis
- `needs-revision` if experiment shows promise but needs changes
- `abandoned` if results indicate approach isn't viable

## API

The SSE server exposes proposal metadata including status:

```bash
curl http://localhost:8001/api/proposals
```

Each proposal includes:
```json
{
  "id": "039-warp-specialized-pingpong-chunkwise-linear-rnn",
  "status": "ongoing",
  "priority": "high",
  "created": "2026-02-15",
  ...
}
```

Filter and sort proposals programmatically using this API.
