# Data

| File | Description |
|---|---|
| `nanolubricant_dataset.csv` | 196 verified experimental datapoints from 60 published studies (the raw extracted data points). |
| `data_sources.csv` | One row per source study: citation key, label, DOI, provenance channel and datapoint count (attribution). |
| `data_dictionary.md` | Description of every column in the dataset. |

Each datapoint is a nanoparticle-lubricant condition paired with its matched
neat-oil baseline, taken from a peer-reviewed source. Every value is traceable to
a specific table, figure or sentence in the cited article (`cof_source`,
`wear_source`, `quote`).

The dataset contains only measured/extracted quantities and their provenance. It
holds no model outputs, predictions or manuscript results; those are regenerated
by the code in `../nanolube/`.

## Attribution

This dataset was compiled from the studies listed in `data_sources.csv`. If you
use it, please credit both this archive (see the top-level `CITATION.cff`) and
the original studies via their DOIs.

## License

The dataset is released under **Creative Commons Attribution 4.0 (CC BY 4.0)** —
see `../LICENSE-DATA.md`. The code is released separately under the MIT License.
