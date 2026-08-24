# Quality review

Review date: 2026-08-16  
Artifact: `report.html`

## Outcome

Pass. The report is decision-oriented, source-audited, responsive, locally reachable, and explicit about what was researched versus what still requires listening tests.

## Content checks

- Leads with one recommendation and preserves five materially different paths.
- Directly answers what comparable teams/models did: hierarchical planning, vocal/accompaniment separation, post-generation voice conversion, short-window re-anchoring, and studio workflows.
- Surveys current released, commercial, specialist, inaccessible, and paper-only candidates without treating them as equally available.
- Separates direct local findings, official capabilities, vendor quality claims, and inference.
- Includes concrete 7-day and 90-day execution plans plus go/no-go gates.
- Includes current hardware reality: one visible RTX 4070 12 GB; MiniMax Music3 is therefore blocked under its official two-GPU setup.
- Treats the Amy-like checkpoint as research-only and defines a consented/original voice path.
- Does not claim that any newly surveyed model was installed, listened to, or independently benchmarked in this research run.

## Evidence checks

- 104 distinct queries recorded.
- 52 source-ledger rows.
- 16 high-impact claims audited.
- 30 numbered report citations.
- Author-run model rankings remain labeled and unresolved.
- LeVo2 was demoted after live repository/model access failed.
- HeartMuLa was corrected from a possible identity engine to a secondary full-song baseline.
- Current Lyria and Eleven pricing and terms were checked against official pages.

## Static checks

- HTML parser: no duplicate IDs.
- Internal anchor check: no missing anchors.
- Local file-link check: no missing targets.
- HTTP check: `200` for both the docs entrypoint and full report.
- Report payload: approximately 33 KB before the final pattern section; no external CSS, fonts, or scripts required.

## Visual QA

Rendered in local Chrome headless and inspected at:

- Desktop: 1440 × 1400 (`report-preview.png`).
- Full-page overview: 1440 × 12000 (`report-full-preview.png`).
- Narrow layout: 500 × 900 (`report-narrow-preview.png`).
- Additional 390 px capture exposed Chrome headless’s minimum-layout-width crop; defensive `max-width`, `min-width`, and horizontal-overflow rules were added. The 500 px responsive capture is clean.

Verified characteristics:

- Sticky navigation remains readable.
- Decision cards are visually dominant.
- Tables are contained in horizontal scrollers.
- Architecture and timeline collapse to one column at narrow widths.
- Color is used consistently for recommended, test, API, and no-go states.
- No observed overlaps, clipped desktop content, broken cards, or illegible type.

## Remaining limitations

- Browser rendering cannot validate the sound-quality judgments; the frozen same-prompt bake-off is still required.
- External URLs can change after the access date.
- Mobile QA below Chrome headless’s effective minimum layout width should be rechecked in a real device/browser if this report becomes public-facing.
