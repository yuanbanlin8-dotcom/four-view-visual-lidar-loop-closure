# Scientific Reports manuscript materials

This edition contains five main figures and three main tables for **Four-View Visual–LiDAR Loop Closure for Drift Correction in Multi-Storey Indoor Mapping**.

- [Manuscript](Manuscript.pdf)
- [Supplementary Information](Supplementary_Information.pdf)
- [Figures](figures/): original PNG exports, plus the editable SVG and vector PDF for the horizontal Figure 2 composite.
- [Tables](tables/): editable CSV exports from the current Word tables.

## Main figures

| Current figure | Content | Earlier asset number |
|---|---|---|
| 1 | Hardware, initial map and system pipeline | 1 |
| 2 | Four-view retrieval, normalization and reverse cross-camera matching | 2 + 3 |
| 3 | Single-storey corridor before/after correction and height-gap profile | 4 |
| 4 | Four-storey map, traversal routes and loop locations | 5 |
| 5 | Sequence sensitivity, floor-height error and floor-factor ablation | 6 |

## Tables

| Current table | Content | Earlier main table(s) |
|---|---|---|
| 1(a) | Single-storey LiDAR-verification ablation | Main 4; formerly S5 in this edition |
| 1(b) | Four-storey camera-input, ranking and LiDAR-verification ablations | 1 + 2 + 5 |
| 1(c) | Single-storey sequence/GNC stress test | Main 6; formerly S6 in this edition |
| 2 | External loop-closure configurations | 3 |
| 3 | NCD Quad-Easy retrieval comparison | 7 |
| S1 | Fixed-camera ranking | Supplementary S2 |
| S2 | Single-storey surface alignment | Supplementary S3 |
| S3 | Implementation parameters | Supplementary S4 |
| S4 | Adjacent-floor heights | Supplementary S5 |

Supplementary Figure S1 contains the expanded NCD retrieval comparison. The former supplementary NCD table is consolidated into main Table 3 and Figure S1.

The existing parent `figures/` and `tables/` directories retain the earlier numbering. This directory provides the current manuscript numbering. Images are copied without recompression; CSV values are taken directly from the current document cells. `SHA256SUMS.json` records the delivered PDFs, figures and tables.

The algorithm release referenced by the manuscript remains commit `288c020630f790a12518f27490ab115965bfbf11`. This update changes manuscript presentation and asset indexing, not algorithm settings or experimental results. Dataset access is described in the repository [download guide](../../DATA_DOWNLOADS.md).

Table 1 contains three blocks with their original metrics and values. The former Tables S5 and S6 are now blocks (a) and (c); Supplementary Tables S1–S4 remain in the supplement.
