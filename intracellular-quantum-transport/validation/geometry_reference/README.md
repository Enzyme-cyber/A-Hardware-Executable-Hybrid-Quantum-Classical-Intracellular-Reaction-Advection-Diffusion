# Reference geometry outputs

These files are the pre-generated topology and schedule audit for the common
three-layer geometry used by the formal configurations:

- 24 conical-frustum subdomains (8 per layer)
- 40 undirected geometric links
- 24 same-layer links
- 8 target-to-upper links
- 8 target-to-lower links
- bidirectional half-ring scheduling with an antipodal meeting sector

They are reference/validation data, not concentration results and not an
additional simulation case. Regenerate the geometry plan with:

```bash
python src/layer_schwarz_qpu_controller.py \
  --config configs/fig6/parameters_1.json \
  --dry-plan
```

The generated geometry files will be written below the output directory defined
by the selected configuration.
