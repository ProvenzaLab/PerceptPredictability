"""Generate the synthetic demo dataset in data/demo/.

Every value is simulated from the seeded parameters below; this script reads no
patient data. The output uses the same file layout and columns as the study data,
so the notebooks can run on it by setting USE_DEMO_DATA = True in config.py.

The simulated results say nothing about the manuscript's findings. The patients
are invented, and the state-dependent parameters were chosen so that the analyses
have a signal to work with.

Usage: python make_demo_data.py
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

OUT_DIR = Path(__file__).parent / 'data' / 'demo'
SEED = 2026
FIRST_DAY, LAST_DAY = -14, 35  # inclusive, 50 days per patient
SAMPLES_PER_DAY = 144  # one sample every 10 minutes
MAX_INTERPOLATED_GAP = 12  # samples (2 h); longer gaps split the contiguous segment

# (lag-1 coefficient of the noise, amplitude of the 24 h rhythm) for each clinical state.
# Higher values make the signal more predictable and raise the LinAR-1 R2.
STATE_PARAMS = {
    'Pre-DBS': (0.66, 0.56),
    'Unknown': (0.60, 0.50),
    'Non-Responder': (0.64, 0.44),
    'Disinhibited': (0.38, 0.37),
    'Transition': (0.45, 0.47),
    'Responder': (0.42, 0.39),
    'Relapse': (0.60, 0.45),
}

STIM_SITES = {
    'BNST': {'model': 'LEAD_B33015', 'cathodes': 'SenSight_1a;Sensight_1b;Sensight_1c', 'contacts': 'ZERO_AND_THREE'},
    'VC/VS': {'model': 'LEAD_3387', 'cathodes': 'FourElectrodes_1', 'contacts': 'ZERO_AND_TWO'},
}

# states: (first day of the state, state), each lasting until the next entry starts.
# ybocs: (day, score); the first entry is the pre-DBS baseline.
# DEMO002 relapses from day 28, with a matching rise in its Y-BOCS score.
PATIENTS = [
    {
        'pt_id': 'DEMO001', 'status': 'Responder', 'site': 'BNST', 'acrophase_h': 15.0, 'phi_offset': 0.02,
        'stim_mA': 5.0, 'pulse_width': 90.0, 'base_power': 950.0,
        'states': [(-14, 'Pre-DBS'), (0, 'Disinhibited'), (3, 'Transition'), (9, 'Responder')],
        'ybocs': [(-30, 38), (-1, 37), (7, 29), (14, 22), (21, 20), (28, 19), (35, 18)],
    },
    {
        'pt_id': 'DEMO002', 'status': 'Responder', 'site': 'VC/VS', 'acrophase_h': 13.0, 'phi_offset': -0.02,
        'stim_mA': 4.5, 'pulse_width': 120.0, 'base_power': 780.0,
        'states': [(-14, 'Pre-DBS'), (0, 'Transition'), (12, 'Responder'), (28, 'Relapse')],
        'ybocs': [(-30, 36), (-1, 36), (7, 30), (14, 21), (21, 20), (28, 30), (35, 32)],
    },
    {
        'pt_id': 'DEMO003', 'status': 'Non-Responder', 'site': 'BNST', 'acrophase_h': 7.0, 'phi_offset': 0.01,
        'stim_mA': 5.5, 'pulse_width': 90.0, 'base_power': 1100.0,
        'states': [(-14, 'Pre-DBS'), (0, 'Unknown'), (1, 'Non-Responder')],
        'ybocs': [(-30, 39), (-1, 39), (7, 38), (14, 37), (21, 37), (28, 36), (35, 36)],
    },
    {
        'pt_id': 'DEMO004', 'status': 'Non-Responder', 'site': 'VC/VS', 'acrophase_h': 9.0, 'phi_offset': -0.01,
        'stim_mA': 4.0, 'pulse_width': 120.0, 'base_power': 860.0,
        'states': [(-14, 'Pre-DBS'), (0, 'Unknown'), (1, 'Non-Responder')],
        'ybocs': [(-30, 35), (-1, 34), (7, 34), (14, 33), (21, 32), (28, 32), (35, 31)],
    },
]

_TIME_NOTE = ("Time of day of the 10-minute BrainSense Timeline sample, rounded down to the "
              "previous 10-minute interval in the patient's home time zone")
_DAY_NOTE = 'Days since DBS initiation (negative values are pre-DBS)'
DATA_DICTIONARY = [
    ('lfp_raw', 'pt_id', 'object', '', 'Patient identifier'),
    ('lfp_raw', 'time_bin_time', 'object', 'HH:MM:SS', _TIME_NOTE),
    ('lfp_raw', 'days_since_dbs', 'int64', 'days', _DAY_NOTE),
    ('lfp_raw', 'lfp_left_raw', 'float64', 'a.u.', 'BrainSense Timeline LFP power in the sensing band (left hemisphere), 10-minute average as reported by the device'),
    ('lfp_raw', 'stim_left', 'float64', 'mA', 'Stimulation amplitude (left hemisphere)'),
    ('lfp_raw', 'lfp_right_raw', 'float64', 'a.u.', 'BrainSense Timeline LFP power in the sensing band (right hemisphere), 10-minute average as reported by the device'),
    ('lfp_raw', 'stim_right', 'float64', 'mA', 'Stimulation amplitude (right hemisphere)'),
    ('lfp_raw', 'left_lead_location', 'object', '', 'DBS target region (left lead)'),
    ('lfp_raw', 'right_lead_location', 'object', '', 'DBS target region (right lead)'),
    ('lfp_raw', 'left_lead_model', 'object', '', 'Medtronic lead model (left lead)'),
    ('lfp_raw', 'right_lead_model', 'object', '', 'Medtronic lead model (right lead)'),
]
for _hem in ('left', 'right'):
    DATA_DICTIONARY += [
        ('lfp_raw', f'active_{_hem}_freq', 'float64', 'Hz', f'Stimulation frequency of the active group ({_hem} hemisphere)'),
        ('lfp_raw', f'active_{_hem}_pulse_width', 'float64', 'µs', f'Stimulation pulse width of the active group ({_hem} hemisphere)'),
        ('lfp_raw', f'active_{_hem}_suspend_amplitude', 'float64', 'mA', f'Amplitude applied when stimulation is suspended ({_hem} hemisphere)'),
        ('lfp_raw', f'active_{_hem}_stim_cathodes', 'object', '', f'Active cathode contacts, semicolon-separated ({_hem} hemisphere)'),
        ('lfp_raw', f'active_{_hem}_stim_anodes', 'object', '', f'Active anode contacts, semicolon-separated ({_hem} hemisphere)'),
        ('lfp_raw', f'active_{_hem}_sensing_contacts', 'object', '', f'Bipolar sensing contact pair ({_hem} hemisphere)'),
        ('lfp_raw', f'active_{_hem}_sensing_frequency', 'float64', 'Hz', f'Center frequency of the sensed band, +/- 2.5 Hz ({_hem} hemisphere)'),
    ]
DATA_DICTIONARY += [
    ('lfp_derived', 'pt_id', 'object', '', 'Patient identifier'),
    ('lfp_derived', 'time_bin_time', 'object', 'HH:MM:SS', _TIME_NOTE),
    ('lfp_derived', 'days_since_dbs', 'int64', 'days', _DAY_NOTE),
    ('lfp_derived', 'lfp_left_OvER_interpolate_z_scored', 'float64', 'SD', 'LFP power after overvoltage event removal, PCHIP interpolation of gaps up to 2 h, and within-day z-scoring (left hemisphere)'),
    ('lfp_derived', 'lfp_right_OvER_interpolate_z_scored', 'float64', 'SD', 'LFP power after overvoltage event removal, PCHIP interpolation of gaps up to 2 h, and within-day z-scoring (right hemisphere)'),
    ('lfp_derived', 'contig_right', 'Int64', '', 'Index of the contiguous recording segment within patient (right hemisphere)'),
    ('lfp_derived', 'contig_left', 'Int64', '', 'Index of the contiguous recording segment within patient (left hemisphere)'),
    ('lfp_derived', 'lfp_left_OvER_interpolate_z_scored_lag_1', 'float64', 'SD', 'Previous sample (t - 10 min) of the z-scored LFP power; LinAR-1 predictor (left hemisphere)'),
    ('lfp_derived', 'lfp_right_OvER_interpolate_z_scored_lag_1', 'float64', 'SD', 'Previous sample (t - 10 min) of the z-scored LFP power; LinAR-1 predictor (right hemisphere)'),
    ('lfp_derived', 'most_recent_YBOCS_value', 'float64', 'Y-BOCS points (0-40)', 'Most recent Y-BOCS score at or before this sample'),
    ('lfp_derived', 'state_label_str', 'object', '', 'Clinical state label (see Methods)'),
    ('lfp_derived', 'response_status', 'object', '', 'Patient-level DBS outcome: Responder, Non-Responder, Unknown/Undetermined'),
    ('ybocs', 'pt_id', 'object', '', 'Patient identifier'),
    ('ybocs', 'YBOCS', 'float64', 'Y-BOCS points (0-40)', 'Y-BOCS total score'),
    ('ybocs', 'days_since_dbs', 'float64', 'days', _DAY_NOTE),
    ('ybocs', 'is_clinical_response', 'bool', '', 'True if YBOCS_reduction >= 0.35'),
    ('ybocs', 'YBOCS_reduction', 'float64', 'fraction', 'Fractional reduction from baseline Y-BOCS (positive = improvement)'),
]


def daily_states(spec, days):
    """Return the clinical state on each day."""
    starts = np.array([start for start, _ in spec['states']])
    labels = np.array([label for _, label in spec['states']], dtype=object)
    return labels[np.searchsorted(starts, days, side='right') - 1]


def simulate_power(spec, states, rng, acrophase_shift=0.0):
    """Simulate one hemisphere's raw power as a 24 h rhythm plus AR(1) noise."""
    n_days = len(states)
    phis = np.array([STATE_PARAMS[s][0] for s in states]) + spec['phi_offset'] + rng.normal(0, 0.07, n_days)
    amps = np.array([STATE_PARAMS[s][1] for s in states])
    phis = np.clip(phis, 0.05, 0.92)

    hours = (np.arange(n_days * SAMPLES_PER_DAY) % SAMPLES_PER_DAY) / 6
    amp_t = np.repeat(amps, SAMPLES_PER_DAY)
    phi_t = np.repeat(phis, SAMPLES_PER_DAY)
    rhythm = amp_t * np.cos(2 * np.pi * (hours - spec['acrophase_h'] - acrophase_shift) / 24)

    noise = np.zeros_like(hours)
    noise_scale = np.sqrt(1 - amp_t ** 2 / 2)  # keeps the total variance near 1
    eps = rng.standard_normal(hours.size)
    for t in range(1, hours.size):
        noise[t] = phi_t[t] * noise[t - 1] + np.sqrt(1 - phi_t[t] ** 2) * eps[t]
    latent = rhythm + noise * noise_scale
    return np.round(np.clip(spec['base_power'] * (1 + 0.3 * latent), 20, None))


def add_gaps(power, rng):
    """Drop some samples: short gaps get interpolated, and one 5 h gap splits the recording."""
    power = power.copy()
    n_days = power.size // SAMPLES_PER_DAY
    for _ in range(12):
        start = rng.integers(SAMPLES_PER_DAY, power.size - 20)
        power[start:start + rng.integers(1, 7)] = np.nan
    long_gap_day = rng.integers(n_days // 2, n_days - 3)
    start = long_gap_day * SAMPLES_PER_DAY + 60
    power[start:start + 30] = np.nan
    return power


def preprocess(power, sample_days):
    """Mirror the derived columns: interpolate short gaps, z-score within day, build lag-1 and segment index."""
    missing = np.isnan(power)
    run_id = np.cumsum(np.r_[True, missing[1:] != missing[:-1]])
    run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()

    filled = power.copy()
    fill_mask = missing & (run_len <= MAX_INTERPOLATED_GAP)
    filled[fill_mask] = PchipInterpolator(np.flatnonzero(~missing), power[~missing])(np.flatnonzero(fill_mask))

    by_day = pd.Series(filled).groupby(sample_days)
    z = ((pd.Series(filled) - by_day.transform('mean')) / by_day.transform('std')).round(6)

    still_missing = z.isna().to_numpy()
    contig = pd.Series(np.cumsum(np.r_[False, still_missing[:-1] & ~still_missing[1:]]), dtype='Int64')
    contig[still_missing] = pd.NA
    return z, contig, z.shift(1)


def ybocs_table(spec):
    days, scores = zip(*spec['ybocs'])
    reduction = 1 - np.array(scores, dtype=float) / scores[0]
    return pd.DataFrame({
        'pt_id': spec['pt_id'],
        'YBOCS': np.array(scores, dtype=float),
        'days_since_dbs': np.array(days, dtype=float),
        'is_clinical_response': reduction >= 0.35,
        'YBOCS_reduction': reduction,
    })


def make_patient(spec, rng):
    days = np.arange(FIRST_DAY, LAST_DAY + 1)
    states = daily_states(spec, days)
    sample_days = np.repeat(days, SAMPLES_PER_DAY)
    sample_states = np.repeat(states, SAMPLES_PER_DAY)
    times = [f'{h:02d}:{m:02d}:00' for h in range(24) for m in range(0, 60, 10)] * days.size
    post_dbs = sample_days >= 0
    site = STIM_SITES[spec['site']]

    raw = pd.DataFrame({'pt_id': spec['pt_id'], 'time_bin_time': times, 'days_since_dbs': sample_days})
    derived = raw.copy()
    # Stimulation ramps up over the first week
    stim = np.where(post_dbs, spec['stim_mA'] * np.clip((sample_days + 1) / 7, 0.4, 1).round(1), 0.0).round(1)

    for hem, shift in (('left', 0.0), ('right', 0.75)):
        power = add_gaps(simulate_power(spec, states, rng, acrophase_shift=shift), rng)
        z, contig, lag = preprocess(power, sample_days)

        raw[f'lfp_{hem}_raw'] = power
        raw[f'stim_{hem}'] = stim
        derived[f'lfp_{hem}_OvER_interpolate_z_scored'] = z
        derived[f'contig_{hem}'] = contig
        derived[f'lfp_{hem}_OvER_interpolate_z_scored_lag_1'] = lag

    for hem in ('left', 'right'):
        raw[f'{hem}_lead_location'] = spec['site']
        raw[f'{hem}_lead_model'] = site['model']
    for hem in ('left', 'right'):
        settings = {
            'freq': 150.0, 'pulse_width': spec['pulse_width'], 'suspend_amplitude': 0.0,
            'stim_cathodes': site['cathodes'], 'stim_anodes': 'Case',
            'sensing_contacts': site['contacts'], 'sensing_frequency': 8.79,
        }
        for name, value in settings.items():
            raw[f'active_{hem}_{name}'] = pd.Series([value] * len(raw)).where(post_dbs)

    yb = ybocs_table(spec)
    recent = yb['days_since_dbs'].searchsorted(sample_days, side='right') - 1
    derived['most_recent_YBOCS_value'] = yb['YBOCS'].to_numpy()[recent]
    derived['state_label_str'] = sample_states
    derived['response_status'] = spec['status']

    raw = raw[[c for f, c, *_ in DATA_DICTIONARY if f == 'lfp_raw']]
    derived = derived[[c for f, c, *_ in DATA_DICTIONARY if f == 'lfp_derived']]
    return raw, derived, yb


def zoned_r2_table(derived, rng):
    """Stand-in for the zoned LinAR-k input: 3-day sliding LinAR-1 R2 plus a little noise."""
    z_col = 'lfp_left_OvER_interpolate_z_scored'
    lag_col = f'{z_col}_lag_1'
    valid = derived.dropna(subset=[z_col, lag_col])
    rows = []
    for day in sorted(valid['days_since_dbs'].unique()):
        train = valid[valid['days_since_dbs'].isin([day - 2, day - 1])]
        test = valid[valid['days_since_dbs'] == day]
        if train['days_since_dbs'].nunique() < 2 or len(test) < SAMPLES_PER_DAY // 2:
            continue
        slope = (train[lag_col] * train[z_col]).sum() / (train[lag_col] ** 2).sum()
        residual = ((test[z_col] - slope * test[lag_col]) ** 2).sum()
        total = ((test[z_col] - test[z_col].mean()) ** 2).sum()
        rows.append((derived['pt_id'].iloc[0], day, round(1 - residual / total + rng.normal(0, 0.02), 6)))
    return pd.DataFrame(rows, columns=['pt_id', 'days_since_dbs', 'r2'])


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, na_rep='n/a', lineterminator='\n')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Same encoding as the study's data dictionary, which the loader expects
    with open(OUT_DIR / 'data_dictionary.tsv', 'w', encoding='cp1252', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['file', 'column', 'dtype', 'units', 'description'])
        writer.writerows(DATA_DICTIONARY)

    participants, zoned = [], []
    for i, spec in enumerate(PATIENTS):
        rng = np.random.default_rng([SEED, i])
        pt = spec['pt_id']
        raw, derived, yb = make_patient(spec, rng)
        write_csv(raw, OUT_DIR / pt / 'ephys' / f'{pt}_lfp_raw.csv')
        write_csv(derived, OUT_DIR / pt / 'ephys' / f'{pt}_lfp_derived.csv')
        write_csv(yb, OUT_DIR / pt / 'behavioral' / f'{pt}_ybocs.csv')
        zoned.append(zoned_r2_table(derived, rng))
        participants.append({
            'Subject ID': pt,
            'Diagnosis': 'Synthetic demo patient (not a real participant)',
            'Sex': 'n/a',
            'Implant Region (Left | Right)': f"{spec['site']} | {spec['site']}",
            'Age at Time of Surgery': 'n/a',
            'Response Status': spec['status'],
            'Baseline Y-BOCS': spec['ybocs'][0][1],
            'Final Y-BOCS': spec['ybocs'][-1][1],
            'Left Lead Model': STIM_SITES[spec['site']]['model'],
            'Right Lead Model': STIM_SITES[spec['site']]['model'],
        })

    write_csv(pd.DataFrame(participants), OUT_DIR / 'participants.csv')
    write_csv(pd.concat(zoned, ignore_index=True), OUT_DIR / 'ark_zoned_r2.csv')
    (OUT_DIR / 'README.txt').write_text(
        'SYNTHETIC DATA: every value here was simulated by make_demo_data.py.\n'
        'These are not recordings from real patients, and results computed from them\n'
        'say nothing about the findings of the manuscript.\n',
        newline='\n',
    )
    size_mb = sum(p.stat().st_size for p in OUT_DIR.rglob('*') if p.is_file()) / 1e6
    print(f'Wrote {len(PATIENTS)} synthetic patients to {OUT_DIR} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
