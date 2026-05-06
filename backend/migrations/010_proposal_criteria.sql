-- Add criteria JSONB column to proposals for structured evaluation contracts
ALTER TABLE proposals ADD COLUMN IF NOT EXISTS criteria JSONB;
