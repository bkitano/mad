-- Add run_number column to experiments table for O(1) rerun lookup
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS run_number INT NOT NULL DEFAULT 1;

-- Backfill existing rows: base experiments get run_number=1, reruns parse from id
UPDATE experiments SET run_number = 1 WHERE id !~ '-r\d+$';
UPDATE experiments SET run_number = CAST(regexp_replace(id, '^.*-r', '') AS INT)
  WHERE id ~ '-r\d+$';
