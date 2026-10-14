# Data attribution

Some results in this repository and on its documentation site are derived
from data that other people published. This file records where those data
come from and the terms that apply to them. The data files themselves are
never committed.

## JMMC OiDB

Calibrated OIFITS files from the JMMC Optical interferometry DataBase
([OiDB](https://oidb.jmmc.fr)) are used under its terms of use
(<https://oidb.jmmc.fr/doc.html>, sections 2, 3 and 9):

- **Licence.** The portal and its contents are under the Creative Commons
  [Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  licence (CC BY-NC-SA 4.0). Figures and numbers derived from OiDB data
  carry the same licence and the attributions below. They are used for
  public astronomical research only.
- **Published data (calibration level L3).** We thank the dataPI (the first
  author of the paper) and cite the original paper.
- **Unpublished data (calibration level L2).** We contact the dataPI before
  presenting any work that uses the data, and record the agreed citation,
  acknowledgement or collaboration policy here. Results from an L2
  collection are not published until that has happened.
- **Acknowledgement.** This research has made use of the Jean-Marie
  Mariotti Center OiDB service, available at <http://oidb.jmmc.fr>.

### Collections

Entries are filled in as each collection is fetched and checked (Stage O0
of `design/plan_oidb.md`); `oidb/manifest.yml` holds the same fields in
machine-readable form.

| Collection id | Data | Level | dataPI | Paper | Terms met |
|---|---|---|---|---|---|
| `782185b2-0727-42b0-a185-b2072732b047` | Gl 229 Ba-Bb, GRAVITY (O1) | L3 | XUAN | Nature 634, 1070, doi:10.1038/s41586-024-08064-x | not yet (thank dataPI, cite paper when used) |
| `696baf06-6c3c-424d-abaf-066c3c324d99` | HR 6819, GRAVITY dynamical masses (binary + Be disk: not clean) (O1) | L3 | Robert Klement | A&A 694, A208, doi:10.1051/0004-6361/202453248 | not yet (thank dataPI, cite paper when used) |
| `647a22a9-5047-4220-ba22-a95047022072` | CHARA/MIRC-X imaging workshop 2023 data (iota Peg, sigma Ori) (O1) | L2 | Gail Schaefer | none (references: Anugu+2020, Schaefer+2016) | cleared by Ben Pope for reduction and display, 2026-10-13; the dataPI contact (who, when, what agreed) is not recorded: add it here; credit the CHARA imaging workshop 2023 data (dataPI Gail Schaefer) |
| `fac164e1-d9d0-4500-8164-e1d9d0450099` | iota Peg, MIRC-X (O1) | L3 | Narsireddy Anugu | AJ 160, 158, doi:10.3847/1538-3881/aba957 | not yet (thank dataPI, cite paper when used) |
| `bda75673-61c6-49f0-a756-7361c699f0c4` | A-star companions, MIRC-X (De Furio+2022) (O1) | L3 | DE FURIO | ApJ 941, 118, doi:10.3847/1538-4357/aca1ad | not yet (thank dataPI, cite paper when used) |
| `f4afc4cd-fd31-40d3-afc4-cdfd3150d340` | HD 45166, GRAVITY (O1) | L3 | DESHMUKH | A&A 695, L20, doi:10.1051/0004-6361/202553692 | not yet (thank dataPI, cite paper when used) |
| `19f7e2cf-2a03-4bb2-b7e2-cf2a03bbb245` | pi1 Gru, PIONIER (O1) | L3 | Claudia Paladini | Nature 553, 310, doi:10.1038/nature25001 | not yet (thank dataPI, cite paper when used) |
| `1e9bab59-54c9-416d-9bab-5954c9416db7` | GG Tau Ab1-Ab2, PIONIER (O2) | L3 | Gaspard Duchene | A&A 686, A188, doi:10.1051/0004-6361/202348755 | not yet (thank dataPI, cite paper when used) |
| `J/A+A/586/A35` | TZ For, PIONIER (O2) | L3 | Gallenne A. | A&A 586, A35, doi:10.1051/0004-6361/201526764 | not yet (thank dataPI, cite paper when used) |
| `371c145f-7889-47d5-9c14-5f7889c7d510` | theta1 Ori C, AMBER (O2) | L3 | KRAUS | A&A 497, 195, doi:10.1051/0004-6361/200810368 | not yet (thank dataPI, cite paper when used) |
| `45840351-f65e-446c-8403-51f65e546c16` | HD 174881, CHARA (O2) | L3 | John Monnier | ApJ 977, 43, doi:10.3847/1538-4357/ad8dcc | not yet (thank dataPI, cite paper when used) |
| `337cfbf7-5327-4875-bcfb-f753270875d0` | Polaris Aa-Ab, MIRC/MIRC-X (O2) | L3 | Gail Schaefer | ApJ 971, 190, doi:10.3847/1538-4357/ad5e7a | not yet (thank dataPI, cite paper when used) |
| `J/A+A/597/A137` | alpha Cen A, B and HD 123999, PIONIER (O2) | L3 | Kervella P. | A&A 597, A137, doi:10.1051/0004-6361/201629505 | not yet (thank dataPI, cite paper when used) |
| `855397ef-dca8-4125-9397-efdca851259d` | kappa Tuc A, MATISSE + GRAVITY (hot dust: not clean) (O2) | L3 | Thomas Stuber | AJ 171, 1, doi:10.3847/1538-3881/adfe66 | not yet (thank dataPI, cite paper when used) |
| `534fbba6-b715-4b41-9278-b6bd0de9f674` | gamma2 Vel, AMBER (wind collision: not clean) (O2) | L3 | Florentin Millour | MNRAS 468, 2655, doi:10.1093/mnras/stx588 | not yet (thank dataPI, cite paper when used) |
| `J/A+A/536/A55` | SS Lep, AMBER + PIONIER (Roche-lobe overflow: not clean) (O2) | L3 | Blind N. | A&A 536, A55, doi:10.1051/0004-6361/201118036 | not yet (thank dataPI, cite paper when used) |

## ESO Science Archive, Phase 3

Reduced GRAVITY products (ESO Phase 3, `ivoa.ObsCore`) of public programmes are read from the ESO Science Archive
Facility and are not committed. They are ESO data products, public once their proprietary period has ended. `oidb/eso_phase3.json` registers each product used by the
gallery: its dp_id, file name, size, SHA-256, header date and programme, and the release date the archive gives
(`scripts/register_eso_phase3.py`).

- **Acknowledgement.** Based on observations collected at the European Southern Observatory under ESO programmes
  111.24UP.001, 111.24UP.003, 111.264P.001 and 113.26HF.001, and obtained from the ESO Science Archive Facility.
- **Cite the paper** whose positions we compare with.

| Data | Programmes | Products | Paper | Terms met |
|---|---|---|---|---|
| ζ Boo, GRAVITY ATs, 4 nights, 2023-04-20 to 2024-05-22 | 111.24UP.001, 111.264P.001, 113.26HF.001 | 4 (105 MB) | Waisberg, Klein & Katz 2025, OJAp 8, 63, doi:10.33232/001c.138625 | public since 2025-05-25 at the latest; cite the paper when used |
| η Oph, GRAVITY ATs, 6 nights, 2023-06-16 to 2024-08-11 | 111.24UP.003, 113.26HF.001 | 6 (134 MB) | Waisberg, Klein & Katz 2025, OJAp 8, 63 | public since 2025-08-11 at the latest; cite the paper when used |
