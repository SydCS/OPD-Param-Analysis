# OPD Delta Analysis

This directory contains a CLI tool for analyzing the final parameter update

```text
Delta W = W_opd - W_src
```

between a source checkpoint and an OPD-trained checkpoint.

Training follows the public implementations and configurations from
[`verl`](https://github.com/verl-project/verl),
[`thunlp/OPD`](https://github.com/thunlp/OPD), and
[`siyan-zhao/OPSD`](https://github.com/siyan-zhao/OPSD), where applicable.
We thank the authors and maintainers of these projects for their efforts.

The analysis entry point is `scripts/analyze_opd_delta.py`.

```bash
python scripts/analyze_opd_delta.py \
  --src_model [W_src] \
  --opd_model [W_opd] \
  --out_dir outputs/opd_delta_analysis \
  --max_exact_svd_dim 4096 \
  --approx_svd_max_numel 3000000
```
