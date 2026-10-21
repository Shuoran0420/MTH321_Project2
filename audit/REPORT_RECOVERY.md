# Report Recovery Record

This is the historical recovery and team-metadata record. The later rubric revision and its newly identified graded-integration issue are documented in `audit/RUBRIC_REVIEW.md`; they supersede any earlier completion wording below.

Recovery date: 2026-10-19 (Asia/Shanghai).

## Source Recovery

The desktop submission archive lacked `report/main.tex` and the compiled report PDF. The report source was recovered by replaying the ten successful report-related patches recorded in the original project session on 2026-10-06. Failed patch attempts were excluded. This recovered the accepted R2 scientific text, the final-candidate title-page wording, the bibliography, and the report evidence map.

The recovered bibliography is byte-identical to the bibliography already in the desktop archive. The recovered five-member source has SHA-256 `e1822377e6986b4c3e16347b0c3b465ad49d01076204c51019afbb10a2772dc9`.

The user's latest confirmed team size is six. The initial recovery restored the `Member 6` contribution placeholder. That six-member source, before the later group-name update below, has SHA-256 `20436acfab67ae4d81069fb72ff8340ac054d3afaf72aa53ec0c5c70f79f8bbc`. The human-completion checklist and final audit placeholder count were updated to six.

The user subsequently confirmed the group name as Group C and supplied the six members in this order: Chuqiao Tang (2360692), Jinyu Ma (2363433), Muqing Xu (2362900), Shuoran Fu (2362626), Taiqi Zhang (2363684), and Yujia Ding (2362094). The report title page, Appendix B member identities, presentation title page, README, final audit, and human-completion checklist now reflect those details. Only the six contribution statements remain for the team to supply. No scientific source text was changed; current checksums are recorded in `audit/FINAL_HASHES.sha256`.

At the user's request, the report's right-hand header, including the cover header, now reads `MTH321`. The remaining report styling was retained.

## Compiled Deliverable

The recovered source was compiled with XeLaTeX and BibTeX using the existing report figures and bibliography. The delivered PDF is `report/final_report.pdf`, with 17 pages. The final LaTeX log has no undefined references, undefined citations, missing figures, or overfull boxes. The converter retains the original page-numbering warning about a repeated `page.1` destination; the report's existing numbering and layout were preserved.

This PDF is a fresh compilation of the recovered source, not a recovered byte-for-byte copy of the old PDF. Its checksum is recorded in `audit/FINAL_HASHES.sha256`; the original PDF's historical checksum is not reused for the rebuilt file.

## Preserved Evidence

No FEM simulation was run. Source code, all numerical CSVs, frozen mesh, existing figures, and numerical logs were retained byte-for-byte from the desktop archive. The presentation changed only to display the confirmed group name and member identities and was recompiled. The original archive remains untouched. The updated file and hash manifests use paths relative to the submission-package root.
