# Paper figures and tables

The current **Scientific Reports** edition contains **five main figures and three main tables**, with six supplementary tables and Figure S1.

Use the [current manuscript and asset index](scientific_reports_20261106/README.md) for PDFs, figures, CSV tables, and the old-to-new numbering map.

The files below retain their earlier numbering so that existing experiment links and editable source figures remain available. In the current edition, former Figures 2 and 3 form the horizontal Figure 2; former Figures 4, 5 and 6 become Figures 3, 4 and 5.

## Earlier source assets


SVGs are extracted from the final manuscript documents, retaining embedded image pixels and editable vectors. CSV tables are exported directly from Word table cells.

| Asset | Subject |
|---|---|
| Figure 1 | Platform, initial mapping and system workflow |
| Figure 2 | Joint four-view retrieval and camera-score normalization |
| Figure 3 | Reverse cross-camera correspondences and corridor revisit |
| Figure 4 | Corridor point clouds and ground-height gap before/after correction |
| Figure 5 | Four-floor routes and loop locations |
| Figure 6 | Sequence sensitivity, adjacent-floor errors and ground-factor ablation |
| Figure S1 | NCD place-recognition comparison |
| Table 1 | Four-view and forward-pair verification stages |
| Table 2 | Four-floor loop counts by retrieval configuration |
| Table 3 | External pipeline accepted loops and cross-floor pairs |
| Table 4 | Single-floor correct/false loop comparison |
| Table 5 | Effect of removing LiDAR verification |
| Table 6 | Sequence/GNC stress ablation |
| Table 7 | NCD retrieval results |
| Table S1 | Expanded NCD retrieval results |
| Table S2 | Single-camera ranking results across floors |
| Table S3 | Ground and structural consistency measurements |
| Table S4 | Algorithm settings and thresholds |
| Table S5 | Mapped adjacent-floor heights and remeasured reference means |

`data/fig2_corridor_ground_gap_by_station.csv` uses the original corridor coordinate. Add 26.5 m for the Figure 4 display coordinate: 31 station centres span 0.25–15.25 m; the axis ends at 15.5 m. Gaps are in metres; multiply by 100 for centimetres.

`data/floor_height_comparison.csv` uses the remeasured adjacent-floor means. `fig6_sequence_gap_data.tsv` retains its source header “Maximum sequence gap”; the manuscript calls it “Maximum query gap”.
