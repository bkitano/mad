# MAD — Launch Blockers / Productization

## User Isolation (Critical) — DONE
- [x] Backend auth middleware — validate Supabase JWT on every API request, extract user ID
- [x] Add `user_id` column to: `chat_sessions`, sandboxes (via Modal tags)
- [x] Scope all queries by user — sandboxes, chats only return the requesting user's resources
- [x] Frontend sends auth token with all API requests
- [x] Volume isolation — user_volumes table tracks ownership, all endpoints check it
- [x] Backfill existing data (volumes, chat sessions) to your user ID
- [ ] Row-level security (RLS) in Supabase as a second layer of defense

## Billing & Subscriptions
- [x] Usage tracking — sandbox_usage table records start/stop per session with GPU type
- [x] Usage endpoint — GET /usage returns active sandboxes + GPU-seconds this month
- [x] Sandbox liveness poller — background task checks Modal every 5min, closes stale sessions
- [ ] Stripe integration — subscription plans with Stripe metered billing
- [ ] Spend limits / quotas — enforce per-plan GPU-hour caps
- [ ] Plan gating — restrict GPU tiers by plan (e.g., free = T4 only, pro = A100)

## Abuse Prevention
- [x] Rate limiting on sandbox creation (30s cooldown between creates)
- [x] Concurrent sandbox cap per user (max 3)
- [x] GPU restriction (T4, L4, A10G only — no A100/H100 without a plan)
- [x] Sandbox timeout capped at 12h

## Account Management
- [ ] User settings page — plan details, usage dashboard, billing portal link
- [ ] Onboarding flow — what does a new user see on first login?
- [ ] Terms of service / acceptable use policy

## Infra / Security
- [x] Backend auth — all endpoints now require valid Supabase JWT
- [x] Volume namespace isolation — ownership table controls access
- [ ] Secrets management — ensure Modal secrets are per-user or properly scoped

## Nice-to-have (not blockers)
- [ ] Email notifications (sandbox terminated, usage approaching limit)
- [ ] Admin dashboard to monitor users/usage
- [ ] Graceful sandbox shutdown with state save on quota exhaustion
