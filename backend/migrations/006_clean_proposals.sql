-- Add proposal_id column to replace filename as the identifier
ALTER TABLE proposals ADD COLUMN IF NOT EXISTS proposal_id TEXT;

-- Backfill proposal_id from filename (strip .md suffix)
UPDATE proposals SET proposal_id = regexp_replace(filename, '\.md$', '') WHERE proposal_id IS NULL;

-- Make proposal_id NOT NULL and unique
ALTER TABLE proposals ALTER COLUMN proposal_id SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_proposals_proposal_id ON proposals(proposal_id);

-- Remove unused columns
ALTER TABLE proposals DROP COLUMN IF EXISTS status;
ALTER TABLE proposals DROP COLUMN IF EXISTS experiment_number;
ALTER TABLE proposals DROP COLUMN IF EXISTS results_file;
ALTER TABLE proposals DROP COLUMN IF EXISTS filename;
