# Spatial aggregation of ROC and PR curves

**Zenodo** : [https://doi.org/10.5281/zenodo.22803022](https://doi.org/10.5281/zenodo.22803022)

Code of "Spatial aggregation of ROC and PR curves" by R. Pic, Z. Zhang, S. Engelke, and J. Ziegel ([arxiv:2609.19517](https://arxiv.org/abs/2609.19517))

## Citation

```bibtex
@misc{Pic2026,
      title={Spatial Aggregation of ROC and Precision-Recall Curves}, 
      author={Romain Pic and Zhongwei Zhang and Sebastian Engelke and Johanna Ziegel},
      year={2026},
      doi={10.48550/arXiv.2609.19517},
}
```

## Article figures

All the code to generate the figures in the article are in `scripts/`.

* Figure 2: `illustration_PAV.py`
* Figure 3: `ce_agg_counts.py`
* Figure 4: `constant-FB_ROC.py`
* Figures 5 and 6: `theoretical_counterexample.py` (and `theoretical_counterexample_nonstationary.py` for the non-stationary case)
* Figures 1 and 7: `plot_agg_roc-pr_pairs.py`
```bash
python scripts/plot_agg_roc-pr_pairs.py \
  --threshold q75 \
  --lead-time 48 \
  --dominance-scenario "hres>graphcast>pangu" \
  --locations "(60.25, -1.5) -- (70.25, -25.0)" \
  --gc-gain-range -10 200 \
  --location-scale-gain-range -5 5 \
  --gain-steps 100000
```
* Figures 8 and 9: `agg_over_regions.py` and `plot_agg_over_regions.py`
    * Fig. 8:
    ```bash
    strategies=(GraphCast location-scale FB parallel)
    lead_times=(48 120 240)

    for strategy in "${strategies[@]}"; do
    for lead_time in "${lead_times[@]}"; do

        python scripts/agg_over_regions.py \
            --threshold "q98" \
            --agg-strategy "${strategy}" \
            --lead-time "${lead_time}" \
            --max-workers "8" \
            --lat-block-size "128" \
            --lat-chunk-size "16" \
            --parallel-backend "loky" \
            --xarray-chunk-lat "0" \
            --xarray-chunk-lon "0" \
            --gc-gain-range "0.8" "4.5" \
            --location-scale-gain-range "-1.5" "1.5" \
            --gain-steps "1000" \
            --summer \
            --land

    done
    done
    ```

    ```bash
    python scripts/plot_agg_over_regions.py \
        --regions world \
        --thresholds q98 \
        --filter land_summer \
        --lead-times 48,120,240
    ```


    * Fig. 9:
    ```bash
    strategies=(GraphCast location-scale FB parallel)
    lead_times=(12 48 120)

    for strategy in "${strategies[@]}"; do
    for lead_time in "${lead_times[@]}"; do

        python scripts/agg_over_regions.py \
            --threshold "max" \
            --agg-strategy "${strategy}" \
            --lead-time "${lead_time}" \
            --max-workers "8" \
            --lat-block-size "128" \
            --lat-chunk-size "16" \
            --parallel-backend "loky" \
            --xarray-chunk-lat "0" \
            --xarray-chunk-lon "0" \
            --gc-gain-range "0.8" "4.5" \
            --location-scale-gain-range "-1.5" "1.5" \
            --gain-steps "1000" \
            --land \
            --no-antarctic
    done
    done
    ```

    ```bash
    python scripts/plot_agg_over_regions.py \
        --regions world \
        --thresholds max \
        --filter land_noant \
        --lead-times 12,48,120
    ```

* Figures 10 and 11: `plot_world_maps.py`

## Additional code and data

The additional code is located in the `tools/` folder.

* `style.py` : common style settings for the figures
* `utils.py` : utilitary functions and parameters

We provide some precomputed/cached data in the following folders:
* `cache/` : to cache large computation results
* `data/` : to store contingency matrices used for global/region aggregated curves 


## Related resources

* [WeatherBench 2](https://weatherbench2.readthedocs.io/)
* [scikit-learn](https://scikit-learn.org/)
