# Configuration map

## Formal Figure 6 cases

The three files in `fig6/` are the manuscript configurations supplied with the
final 3-D Schwarz package. Their original filenames are retained for direct
traceability to the archived runs.

| Configuration | Role in the package | Distinguishing input/settings |
| --- | --- | --- |
| `parameters_1.json` | Fig. 6 formal case 1 | Root-cone initial input; direct surface-input events disabled |
| `parameters_side.json` | Fig. 6 formal case 2 | One-sided membrane source selected over 0-75 degrees |
| `parameters_wholeside.json` | Fig. 6 formal case 3 | Wrap-around membrane selector (345-60 degrees) with stronger relaxation/interface-transfer settings |

All three formal cases use the full compact grid (`nz=6`, radial ring counts
`[1,4,8,12]`), two Schwarz iterations, QPanda real-QPU dispatch, 500 shots,
and a 24-subdomain three-layer geometry.

## Smoke configuration

`smoke/layer_geometry_smoke.json` is a fast classical integration test. It uses
a shorter time window, a reduced axial grid, no organelles, one Schwarz
iteration, and `classical_mock` dispatch. It is not a fourth scientific case and
must not be used as a source for the manuscript's quantitative results.
