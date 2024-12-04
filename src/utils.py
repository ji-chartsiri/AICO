import numpy as np
import pandas as pd

def process_vars(all_vars, ignored_vars, discrete_vars, categorical_vars):
    """
    Process the variables in the dataset and categorize them as continuous, discrete, or categorical.

    Parameters:
    - all_vars (list or array): List of all variable names in the dataset.
    - ignored_vars (list or array): List of variable names to ignore.
    - discrete_vars (list or array): List of discrete variable names.
    - categorical_vars (list or dict): If list, it should contain prefixes of categorical variables (e.g., "color" will match
      columns like ["color_blue", "color_yellow"]). If dict, it should map categorical variable names to a list of their dummy variables.

    Returns:
    - continuous_vars (np.array): Array of continuous variable names.
    - discrete_vars (np.array): Array of discrete variable names.
    - categorical_vars (dict): Dictionary mapping categorical variable names to their dummy variables.
    - all_vars (pd.DataFrame): DataFrame containing all variables with their types (continuous, discrete, categorical, or ignored).
    """
    ignored_vars = np.array(ignored_vars, dtype=str)
    all_vars = np.setdiff1d(np.array(all_vars, dtype=str), ignored_vars)
    discrete_vars = np.array(discrete_vars, dtype=str)

    if isinstance(categorical_vars, dict):
        for var in categorical_vars:
            categorical_vars[var] = np.array(categorical_vars[var], dtype=str)
    elif isinstance(categorical_vars, list):
        temp = dict()
        for var in categorical_vars:
            temp[var] = all_vars[np.char.startswith(all_vars, var)]
        categorical_vars = temp

    flatten_categorical_vars = np.concatenate([np.array(vars) for vars in categorical_vars.values()])
    continuous_vars = all_vars[~np.isin(all_vars, np.union1d(discrete_vars, flatten_categorical_vars))]

    all_vars = (pd.concat([pd.DataFrame(dict(variable=continuous_vars, type='continuous')),
                           pd.DataFrame(dict(variable=discrete_vars, type='discrete')),
                           pd.DataFrame(dict(variable=categorical_vars.keys(), type='categorical')),
                           pd.DataFrame(dict(variable=ignored_vars, type='ignored'))])
                .set_index('variable'))

    return continuous_vars, discrete_vars, categorical_vars, all_vars

def summary(result, alpha, sample_size, score_func):
    """
    Print a summary of the AICO test results.

    Parameters:
    - result (pd.DataFrame): DataFrame containing the test results for each variable.
    - alpha (float): Significance level used in the test.
    - sample_size (int): Number of samples used in the test.
    - score_func (callable): Score function used to evaluate model performance.
    """
    def format_number(num, is_coverage=False):
        """Formats numbers: scientific notation for |num| < 1e-3, otherwise fixed-point with 3 decimals,
        but with 5 decimals for coverage."""
        if is_coverage:
            return f'{num:.5f}'  # Format with 5 decimals for coverage
        return f'{num:.3e}' if num != 0 and abs(num) < 1e-3 else f'{num:.3f}'

    def format_table(df, columns, col_widths):
        """Formats a DataFrame into a table with specified column widths."""
        if df.empty:
            return '* No variables'
        # Modify the column names for display purposes
        display_columns = [f'[{col}' if col == 'lower' else f'{col}]' if col == 'upper' else col for col in columns]
        header = ' '.join(f'{col:<{w}}' for col, w in zip(display_columns, col_widths))
        rows = [
            ' '.join(
                f'{format_number(row[col], is_coverage=(col == "coverage")) if col in num_cols else str(row[col]):<{w}}'
                for col, w in zip(columns, col_widths)
            )
            for _, row in df.iterrows()
        ]
        return f'{header}\n{"-" * len(header)}\n' + '\n'.join(rows)

    # Title and header
    header = '=' * 92
    summary = []
    summary.append('================================[AICO Test Result Summary]==================================')
    summary.append(f'{"alpha:":<20}{alpha:<35}{"# testing samples:":<25}{sample_size:,}')
    summary.append(f'{"# tested variables:":<20}{(result["type"] != "ignored").sum():<35}{"# significant variables:":<25}{result["significance"].sum()}')
    summary.append(f'{"score function:":<20}{score_func.__name__:<35}')
    summary.append(header)

    # Map types and ensure proper formats
    type_mapping = {'continuous': 'con', 'discrete': 'dis', 'categorical': 'cat'}
    result['type'] = result['type'].map(type_mapping).fillna(result['type'])
    result['rank'] = result['rank'].fillna(-1).astype(int)

    # Columns and widths for tables (widened for full visibility)
    columns = ['rank', 'variable', 'median', 'p_value', 'lower', 'upper', 'coverage']
    col_widths = [6, 20, 12, 12, 12, 12, 12]  # Increased column widths to fit content
    num_cols = {'median', 'p_value', 'lower', 'upper', 'coverage'}

    # Separate and format tables
    significant_vars = result[result['significance'] == 1].sort_values(by='rank')
    insignificant_vars = result[result['significance'] == 0].sort_values(by='rank')

    summary.append('[Significant Variables]')
    summary.append(format_table(significant_vars, columns, col_widths))
    summary.append(header)

    summary.append('[Insignificant Variables]')
    summary.append(format_table(insignificant_vars, columns, col_widths))
    summary.append(header)

    summary.append('[Ignored Variables]')
    ignored_vars = result[result['type'] == 'ignored']['variable'].tolist()
    summary.append('\n'.join([f' - {var}' for var in ignored_vars]) if ignored_vars else '* No ignored variables')
    summary.append(header)

    print('\n'.join(summary))

