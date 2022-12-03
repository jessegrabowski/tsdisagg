import pandas as pd

MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
YEARLY_FREQS = ['A', 'BA', 'AS', 'BAS']
QUARTERLY_FREQS = ['Q', 'BQ', 'QS', 'BQS']

VALID_YEARLY = YEARLY_FREQS + [f'{freq}-{month}' for freq in YEARLY_FREQS for month in MONTHS]
VALID_QUARTERLY = QUARTERLY_FREQS + [f'{freq}-{month}' for freq in QUARTERLY_FREQS for month in MONTHS]
VALID_MONTHLY = ['M', 'MS', 'BM', 'BMS']

MONTHS_IN_YEAR = 12
MONTHS_IN_QUARTER = 3
QUARTERS_IN_YEAR = 4

FREQ_CONVERSION_FACTORS = {
    'yearly': {
        'monthly': MONTHS_IN_YEAR,
        'quarterly': QUARTERS_IN_YEAR
    },
    'quarterly': {
        'monthly': MONTHS_IN_QUARTER
    }
}

OFFSET_CONVERSIONS = {
    'yearly': {
        'monthly': {'months': MONTHS_IN_YEAR - 1},
        'quarterly': {'months': (QUARTERS_IN_YEAR - 1) * MONTHS_IN_QUARTER}
    },
    'quarterly': {
        'monthly': {'months': MONTHS_IN_QUARTER - 1}
    }
}

LONG_FREQ_TO_CODE = {'yearly': 'A', 'quarterly': 'Q', 'monthly': 'M'}
CODE_TO_LONG_FREQ = {v: k for k, v in LONG_FREQ_TO_CODE.items()}

FREQ_TO_ORDER = {
    'yearly': 10,
    'quarterly': 9,
    'monthly': 8
}

ORDER_TO_FREQ = {v: k for k, v in FREQ_TO_ORDER.items()}


def is_annual_freq(freq_str):
    return freq_str in VALID_YEARLY


def is_quarterly_freq(freq_str):
    return freq_str in VALID_QUARTERLY


def is_monthly_freq(freq_str):
    return freq_str in VALID_MONTHLY


VALIDATE_FUNCS = [is_annual_freq, is_quarterly_freq, is_monthly_freq]


def validate_freqs(*freqs):
    for freq in freqs:
        if not any([f(freq) for f in VALIDATE_FUNCS]):
            raise NotImplementedError(f'Only annual, quarterly and monthly frequencies are supported, found {freq}')


def get_frequency_name(freq):
    if is_annual_freq(freq):
        return 'yearly'
    if is_quarterly_freq(freq):
        return 'quarterly'
    if is_monthly_freq(freq):
        return 'monthly'

    return ''


def auto_step_down_base_freq(freq):
    if not isinstance(freq, str):
        freq = freq.name

    freq_name = get_frequency_name(freq)
    one_freq_down = FREQ_TO_ORDER[freq_name] - 1
    high_freq_name = ORDER_TO_FREQ.get(one_freq_down)

    if not high_freq_name:
        raise NotImplementedError(f'No frequency lower than {freq_name} currently supported')

    low_freq_code = LONG_FREQ_TO_CODE[freq_name]
    high_freq_code = LONG_FREQ_TO_CODE[high_freq_name]

    base, suffix = freq.split('-')
    new_base = base.replace(low_freq_code, high_freq_code)
    if high_freq_name not in ['yearly', 'quarterly']:
        return new_base

    return new_base + '-' + suffix


def get_high_freq_endpoints(endpoint_date, low_freq_name, high_freq_name, target_freq, mode='start'):
    freq_base = target_freq.split('-')[0]

    # If it's a start-of-month business calendar, it can end up that the high_freq_start is not the correct date
    # e.g. if the 1st of the low_freq_start month happens to be a weekend. Need to shift it back to the true
    # first non-weekend date
    if mode == 'start':
        offset = pd.DateOffset(**OFFSET_CONVERSIONS[low_freq_name][high_freq_name])
        high_freq_start = endpoint_date - offset

        if 'S' in freq_base and 'B' in freq_base:
            start_day = high_freq_start.day
            start_date = high_freq_start.isocalendar().weekday

            if start_day != 1:
                # Case 1: The 1st of the month is not a weekend. We go to the first.
                if (start_date - start_day + 1) not in [0, 7]:
                    high_freq_start -= pd.DateOffset(days=start_day - 1)

                # Case 2: The 1st of the month is a weekend. Shift to the first monday.
                else:
                    shift = min(max(start_day - 1, 1), 6)
                    high_freq_start -= pd.DateOffset(days=shift)

        # If we're dealing with a business calendar, check that the offset result is not on the weekend
        # and shift back to Friday if so.
        elif 'B' in freq_base:
            start_day = high_freq_start.isocalendar().weekday
            days_to_friday = max(0, start_day - 5)
            high_freq_start -= pd.DateOffset(days=days_to_friday)
        return high_freq_start

    if mode == 'end':
        offset = pd.DateOffset(**OFFSET_CONVERSIONS[low_freq_name][high_freq_name])
        high_freq_end = endpoint_date + offset
        if 'S' in freq_base and 'B' in freq_base:
            end_day = high_freq_end.day
            end_date = high_freq_end.isocalendar().weekday

            if end_day != 1:
                if (end_date - end_date - 1) not in [0, 7]:
                    high_freq_end += pd.DateOffset(days=end_day + 1)
                else:
                    shift = max(min(end_day + 1, 1), 6)
                    high_freq_end += pd.DateOffset(days=shift)
        return high_freq_end


def get_last_day(month, year):
    if month == 2:
        if year % 4 == 0:
            return 29
        return 28
    if month in [4, 6, 9, 11]:
        return 30

    return 31


def business_cal_adjust(date, adjust_forward=True):
    day = date.day
    weekday = date.weekday()

    if not adjust_forward:
        if day != 1:
            # Case 1: The 1st of the month is not a weekend. We go to the first.
            if (weekday - day + 1) not in [0, 6]:
                date -= pd.DateOffset(days=day - 1)

            # Case 2: The 1st of the month is a weekend. Shift to the first monday.
            else:
                shift = min(max(day - 1, 1), 5)
                date -= pd.DateOffset(days=shift)

    else:
        month = date.month
        year = date.year

        last_day = get_last_day(month, year)
        delta = last_day - day
        last_weekday = (weekday + delta) % 7
        if day != last_day:
            # Case 1: The last of the month is not a weekend. Go to the last.
            if last_weekday not in [5, 6]:
                date += pd.DateOffset(days=delta)

            # Case 2: The last day of the month is a weekend. Go to the last friday.
            else:
                shift = last_weekday - 4
                date += pd.DateOffset(days=delta - shift)

    return date


def get_frequency_names(df, target_freq):
    low_freq = df.index.freq or df.index.inferred_freq

    low_freq_name = get_frequency_name(low_freq)
    high_freq_name = get_frequency_name(target_freq)

    return low_freq_name, high_freq_name


def make_names_from_frequencies(df, target_freq):
    low_freq_name, high_freq_name = get_frequency_names(df, target_freq)

    var_name = ''
    if isinstance(df, pd.Series):
        var_name = df.name or 'data'
    elif isinstance(df, pd.DataFrame):
        var_name = df.columns[0]

    return var_name, f'{low_freq_name}_{var_name}', f'{high_freq_name}_{var_name}'


def make_companion_index(df, target_freq):
    low_freq = df.index.freq or df.index.inferred_freq
    low_freq_name, high_freq_name = get_frequency_names(df, target_freq)

    if not FREQ_TO_ORDER[low_freq_name] > FREQ_TO_ORDER[high_freq_name]:
        raise ValueError(f'target_freq must be of higher frequency than the frequency on the data. Found '
                         f'target_freq {target_freq}, which is {high_freq_name}, while data is {low_freq.name}, which is '
                         f'{low_freq_name}.')

    low_freq_df = df.copy()
    start_date, end_date = low_freq_df.index[[0, -1]]

    offset = pd.DateOffset(**OFFSET_CONVERSIONS[low_freq_name][high_freq_name])
    base = target_freq.split('-')[0]

    if 'S' in base:
        end_date += offset
    else:
        start_date -= offset

    if 'B' in base:
        start_date = business_cal_adjust(start_date, adjust_forward=False)
        end_date = business_cal_adjust(end_date, adjust_forward=True)

    high_freq_index = pd.date_range(start=start_date,
                                    end=end_date,
                                    freq=target_freq)

    high_freq_index.freq = target_freq
    return high_freq_index


def align_and_merge_dfs(low_freq_df, high_freq_df):
    low_freq = low_freq_df.index.inferred_freq

    # these are stored as adverbs (e.g. yearly), so remove the -ly suffix
    attr = get_frequency_name(low_freq)[:-2]

    low_freq_idx = getattr(low_freq_df.index, attr)
    high_freq_idx = getattr(high_freq_df.index, attr)
    low_freq_set = set(low_freq_idx)
    high_freq_set = set(high_freq_idx)

    full_set = sorted(list(low_freq_set.union(high_freq_set)))
    C_mask = [x in low_freq_idx for x in full_set]

    df = pd.merge(low_freq_df, high_freq_df, left_index=True, right_index=True, how='outer')

    return df, C_mask
