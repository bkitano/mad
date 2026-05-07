-- Add criteria JSONB column to proposals for harness evaluation contracts.
-- criteria stores an ExperimentCriteria object: tasks with metric thresholds,
-- time budget, allowed/forbidden edit paths, and baseline config.
ALTER TABLE proposals ADD COLUMN IF NOT EXISTS criteria JSONB;
