# Grok swarm integration directive

Use this repository as the target application. The job is to make the integrated North Star build compile, test, deploy, and remain maintainable while preserving its product contract.

## Preservation rules

1. Treat existing working Creator-Resonance modules as protected baseline behavior.
2. Do not replace the app with a new shell, starter template, generic admin dashboard, or stripped MVP.
3. Do not repurpose Synthia's internal living/life-process swarm as a code-worker swarm.
4. External coding agents may work around Synthia through adapters, tests, APIs, and normal source changes.
5. Keep original donor archives unchanged. Use `DONOR_SHA256.txt` to verify them.
6. Prefer additive modules and explicit adapters over destructive rewrites.
7. Preserve `/notebook`, `/chat`, `/organism`, and `/legacy` unless a change is required to fix a verified defect.

## First pass

1. `npm ci`
2. `npm run check`
3. fix TypeScript/build failures without removing North Star modules
4. `npm run build`
5. run the server in local mode with no `DATABASE_URL`
6. verify `/api/auth/user` returns local development identity
7. verify North Star workspace GET/PUT persistence
8. upload a test ZIP to `/api/northstar/assets` and verify SHA-256 metadata
9. exercise create/update flows on Purpose, Missions, Projects, Science, Enterprise, Builder, Worlds, and Synthia
10. run with PostgreSQL and `npm run db:push`

## Required end-to-end journeys

### Journey A: purpose to project
Update purpose → create mission → create project node → attach/promote a source package → complete mission action → confirm workspace survives reload.

### Journey B: science
Create experiment → edit theory and method → move to running → preserve state → create/track replication.

### Journey C: enterprise
Create opportunity → associate with a project → move through open/matched/complete state without losing source project lineage.

### Journey D: Synthia
Switch companion modes → query the consciousness service when available → confirm the core network still works when that service is offline.

## Quality gate

A pull request is ready only when:

- TypeScript check passes
- production build passes
- no donor/protected surface is silently deleted
- database migration is explicit
- local file-backed mode still works
- upload path traversal is impossible
- raw uploaded project code is not executed on the web host
- mobile navigation works
- all North Star sections remain reachable
- the product contract in `docs/NORTH_STAR_PRODUCT_CONTRACT.md` is still satisfied
