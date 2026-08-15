# 19 — Product KPI computed from Workbench activity

**What to build:** The operator sees the product KPI — corrections per 100 symbols and reviewer minutes per accepted Sheet — computed from Review Workbench activity, so "the software works" is measurable on real corpus Sheets. Together with the component gates, this is the definition of "works" that triggers hazop-ai reintegration.

**Blocked by:** 11 (Workbench queues and review progress).

**Status:** ready-for-agent

- [ ] Corrections per 100 symbols is computed from verdict records (reject + edit counted as corrections against symbols reviewed), reportable per Document and per Convention Profile.
- [ ] Reviewer minutes per accepted Sheet is computed from Workbench activity timing and Sheet review completion, with the measurement basis stated alongside the number.
- [ ] The KPI is visible to the operator (in the Workbench or its report output) and recomputable from persisted verdicts and activity — not session memory.
- [ ] KPI output references Sheets and Documents by identifier only; no drawing content leaves the local store per ADR-0001.
- [ ] Tests compute KPIs from prepared verdict/activity fixtures with known expected values, under the offline invariant.

## Comments
