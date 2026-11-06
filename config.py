"""Settings shared by the notebooks, including the switch for the demo dataset."""
import os
from pathlib import Path

# Set to True to run the notebooks on the synthetic data in data/demo/, then restart the kernel.
# Demo outputs go to separate folders, so they never overwrite the real figures and tables.
USE_DEMO_DATA = False

# Optional overrides for automated test runs
if 'NP_USE_DEMO_DATA' in os.environ:
    USE_DEMO_DATA = os.environ['NP_USE_DEMO_DATA'] == '1'
_OUTPUT_ROOT = Path(os.environ.get('NP_OUTPUT_ROOT', '.'))

if USE_DEMO_DATA:
    DATA_ROOT = Path('data/demo')
    FIG_DIR = _OUTPUT_ROOT / 'figures' / 'demo'
    TABLE_DIR = _OUTPUT_ROOT / 'tables' / 'demo'
    DERIVED_DIR = _OUTPUT_ROOT / 'data' / 'demo_derived'

    EXAMPLE_R_PT, EXAMPLE_NR_PT = 'DEMO001', 'DEMO003'
    RELAPSE_EXAMPLE_PT = 'DEMO001'
    RELAPSE_TO_NR_PT = 'DEMO002'
    RELAPSE_EXAMPLE_PERIODS = [(9, 16, 'Responder'), (16, 22, 'Relapse'),
                               (22, 28, 'Partial Responder'), (28, 36, 'Relapse')]
    FIG1_SEGMENT_BOUNDS = [(None, 10), (10, 25), (25, None)]
    FIG1_DAY_RANGES = [(-7, -4), (15, 18)]
    VIDEO_LAST_DAY = 35
    N_PERMUTATIONS = 200
else:
    DATA_ROOT = Path('data/DABI_upload')
    FIG_DIR = _OUTPUT_ROOT / 'figures'
    TABLE_DIR = _OUTPUT_ROOT / 'tables'
    DERIVED_DIR = _OUTPUT_ROOT / 'data'

    EXAMPLE_R_PT, EXAMPLE_NR_PT = 'B011', 'B006'
    RELAPSE_EXAMPLE_PT = 'B001'
    RELAPSE_TO_NR_PT = 'B014'
    RELAPSE_EXAMPLE_PERIODS = [(54, 239, 'Responder'), (239, 292, 'Relapse'),
                               (292, 378, 'Partial Responder'), (378, 462, 'Relapse')]
    FIG1_SEGMENT_BOUNDS = [(None, 100), (100, 250), (250, None)]
    FIG1_DAY_RANGES = [(-21, -18), (185, 188)]
    VIDEO_LAST_DAY = 500
    N_PERMUTATIONS = 10000

for _folder in (FIG_DIR, TABLE_DIR, DERIVED_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

if USE_DEMO_DATA:
    print('DEMO MODE: using synthetic data from data/demo/. Results say nothing about the manuscript\'s findings.')
    print(f'Outputs go to {FIG_DIR}, {TABLE_DIR} and {DERIVED_DIR}.')
