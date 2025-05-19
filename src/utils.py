import numpy as np
import pandas as pd

def process_vars(vars, vars_ignored, vars_discrete, vars_categorical):
    """
    Process the variables in the dataset and categorize them as continuous, discrete, or categorical.

    Parameters:
    - vars (list or array): List of all variable names in the dataset.
    - vars_ignored (list or array): List of variable names to ignore.
    - vars_discrete (list or array): List of discrete variable names.
    - vars_categorical (list or dict): If list, it should contain prefixes of categorical variables (e.g., "color" will match
      columns like ["color_blue", "color_yellow"]). If dict, it should map categorical variable names to a list of their dummy variables.

    Returns:
    - vars (pd.DataFrame): DataFrame containing all variables with their types (continuous, discrete, categorical, or ignored) and columns in original dataframe.
    """
    vars = np.array(vars, dtype=str)
    vars_index_original = (pd.DataFrame(dict(variable=vars))
                           .reset_index(names='variable_index')
                           .set_index('variable'))

    vars_ignored = np.array(vars_ignored, dtype=str)
    vars = np.setdiff1d(vars, vars_ignored)
    vars_discrete = np.array(vars_discrete, dtype=str)

    if isinstance(vars_categorical, dict):
        for var in vars_categorical:
            vars_categorical[var] = np.array(vars_categorical[var], dtype=str)
    elif isinstance(vars_categorical, list):
        temp = dict()
        for var in vars_categorical:
            temp[var] = vars[np.char.startswith(vars, var)]
        vars_categorical = temp

    flatten_vars_categorical = np.concatenate([np.array(vars) for vars in vars_categorical.values()])
    vars_continuous = vars[~np.isin(vars, np.union1d(vars_discrete, flatten_vars_categorical))]

    vars = (pd.concat([pd.DataFrame(dict(variable=vars_continuous, 
                                         type='continuous', 
                                         columns=[[col] for col in vars_continuous])),
                       pd.DataFrame(dict(variable=vars_discrete, 
                                         type='discrete', 
                                         columns=[[col] for col in vars_discrete])),
                       pd.DataFrame(dict(variable=vars_categorical.keys(), 
                                         type='categorical', 
                                         columns=[vars_categorical[var] for var in vars_categorical.keys()])),
                       pd.DataFrame(dict(variable=vars_ignored, 
                                         type='ignored', 
                                         columns=[[col] for col in vars_ignored]))])
            .set_index('variable'))
    vars['variable_index'] = vars.apply(lambda var: vars_index_original.loc[var['columns'], 'variable_index'].min(), axis=1)

    return vars

def summary(result):
    """
    Fully dynamic AICO test summary, with two-column high-level info, configurable
    super-headers, and column order determined by the super-headers.
    """

    # ------------------------ 1) High-Level Info (two columns) ------------------------
    alpha        = result['alpha'].iloc[0]
    sample_size  = int(result['sample_size'].iloc[0])
    tested_count = (result['type'] != 'ignored').sum()
    sig_count    = result['significance'].sum()
    score_func   = result['score_func'].iloc[0]
    seed         = result['seed'].iloc[0]
    prob_os      = result['prob_os'].iloc[0]

    info_lines = [
        ("alpha:", alpha),
        ("# testing samples:", f"{sample_size:,}"),
        ("# tested variables:", tested_count),
        ("# significant variables:", sig_count),
        ("score function:", score_func),
        ("seed:", f"{seed} (realized)" if seed is not None else "randomized (unrealized)"),
    ]

    # ------------------------ 2) Columns + Super-Header Definitions ------------------------
    #
    # Instead of a list, columns_config is now a dict {column_key: {header, width, ...}}.
    # Order does NOT matter here; we’ll rely on super_headers to dictate the final order.
    columns_config = {
        "rank":             {"header": "rank",     "width": 6},
        "variable":         {"header": "variable", "width": 20},
        "median":           {"header": "median",   "width": 12},
        "prob_reject":      {"header": "P(reject)",   "width": 12},
        "p_value":          {"header": "p-value",  "width": 12},
        "p_value_lower":    {"header": "lower",  "width": 12},
        "p_value_upper":    {"header": "upper",  "width": 12},
        "lower_os":         {"header": "lower",    "width": 12},  # one-sided lower
        "lower_os_1":       {"header": f"{100*(1-prob_os):.1f}%",    "width": 12},  # one-sided lower
        "lower_os_2":       {"header": f"{100*prob_os:.1f}%",    "width": 12},  # one-sided lower
        "coverage_os":      {"header": "coverage", "width": 12},  # one-sided coverage
        "lower_ts":         {"header": "lower",    "width": 12},  # two-sided lower
        "upper_ts":         {"header": "upper",    "width": 12},
        "coverage_ts":      {"header": "coverage", "width": 12},
    }

    # Each super-header block: "title", "columns" (list of column_keys), "align".
    # The final table will include columns in exactly this order (left→right, block by block).
    if seed is None:
        super_headers_significant = [
            {"title": "[Significant Variables]",        "columns": ["rank", "variable"],                           "align": "left"},
            {"title": "",                               "columns": ["median", "prob_reject"],                      "align": "left"},
            {"title": "P-Value Distribution",           "columns": ["p_value_lower", "p_value_upper"],             "align": "left"},
            {"title": "Randomized One-Sided CI",        "columns": ["lower_os_1", "lower_os_2", "coverage_os"],    "align": "left"},
            {"title": "Non-Randomized Two-Sided CI",    "columns": ["lower_ts", "upper_ts", "coverage_ts"],        "align": "left"}
        ]
        super_headers_inconclusive = [
            {"title": "[Inconclusive Variables]",       "columns": ["rank", "variable"],                           "align": "left"},
            {"title": "",                               "columns": ["median", "prob_reject"],                      "align": "left"},
            {"title": "P-Value Distribution",           "columns": ["p_value_lower", "p_value_upper"],             "align": "left"},
            {"title": "Randomized One-Sided CI",        "columns": ["lower_os_1", "lower_os_2", "coverage_os"],    "align": "left"},
            {"title": "Non-Randomized Two-Sided CI",    "columns": ["lower_ts", "upper_ts", "coverage_ts"],        "align": "left"}
        ]
        super_headers_insignificant = [
            {"title": "[Insignificant Variables]",      "columns": ["rank", "variable"],                           "align": "left"},
            {"title": "",                               "columns": ["median", "prob_reject"],                      "align": "left"},
            {"title": "P-Value Distribution",           "columns": ["p_value_lower", "p_value_upper"],             "align": "left"},
            {"title": "Randomized One-Sided CI",        "columns": ["lower_os_1", "lower_os_2", "coverage_os"],    "align": "left"},
            {"title": "Non-Randomized Two-Sided CI",    "columns": ["lower_ts", "upper_ts", "coverage_ts"],        "align": "left"}
        ]
    else:
        super_headers_significant = [
            {"title": "[Significant Variables]",        "columns": ["rank", "variable"],                           "align": "left"},
            {"title": "",                               "columns": ["median"],                                     "align": "left"},
            {"title": "Realized",                       "columns": ["p_value"],                                    "align": "left"},
            {"title": "Realized One-Sided CI",          "columns": ["lower_os", "coverage_os"],                    "align": "left"},
            {"title": "Non-Randomized Two-Sided CI",    "columns": ["lower_ts", "upper_ts", "coverage_ts"],        "align": "left"}
        ]

        super_headers_insignificant = [
            {"title": "[Insignificant Variables]",      "columns": ["rank", "variable"],                           "align": "left"},
            {"title": "",                               "columns": ["median"],                                     "align": "left"},
            {"title": "Realized",                       "columns": ["p_value"],                                    "align": "left"},
            {"title": "Realized One-Sided CI",          "columns": ["lower_os", "coverage_os"],                    "align": "left"},
            {"title": "Non-Randomized Two-Sided CI",    "columns": ["lower_ts", "upper_ts", "coverage_ts"],        "align": "left"}
        ]
    

    def get_ordered_columns(super_headers, col_config):
        """
        Flatten the list of columns from super_headers in order (left→right),
        building a list of { 'key', 'header', 'width' } dicts from col_config.
        """
        ordered = []
        for block in super_headers:
            for col_key in block["columns"]:
                # Look up header & width from columns_config dict
                ordered.append({
                    "key":    col_key,
                    "header": col_config[col_key]["header"],
                    "width":  col_config[col_key]["width"]
                })
        return ordered

    # ------------------------ 3) Helper Functions ------------------------
    def format_number(x, coverage=False):
        """Coverage columns get 5 decimals; otherwise scientific or .3f."""
        if coverage:
            return f"{x:.5f}"
        return f"{x:.3e}" if (x != 0 and abs(x) < 1e-3) else f"{x:.3f}"

    def make_banner_line(columns_list):
        """
        Compute a banner line length from the sum of the columns' widths, or at least 70.
        """
        total_width = sum(c["width"] for c in columns_list)
        return "=" * max(total_width, 70)

    def print_two_column_info(info_items, columns_list):
        """Print the high-level info items in two columns side by side."""
        banner = make_banner_line(columns_list)
        line_width = len(banner)
        col_width = line_width // 2

        # Group info items in pairs
        pairs = [info_items[i:i+2] for i in range(0, len(info_items), 2)]
        for pair in pairs:
            if len(pair) == 2:
                (lab1, val1), (lab2, val2) = pair
                left_str  = f"{lab1:<25}{val1}"
                right_str = f"{lab2:<25}{val2}"
                print(f"{left_str:<{col_width}}{right_str:<{col_width}}")
            else:
                lab1, val1 = pair[0]
                single_str = f"{lab1:<25}{val1}"
                print(f"{single_str:<{col_width}}")

    def print_info(info, columns_list):
        """Print top banner, summary title, then info in two columns, then another banner."""
        banner = make_banner_line(columns_list)
        print(banner)
        print("[AICO Test Result Summary]".center(len(banner)))
        print(banner)
        print_two_column_info(info, columns_list)
        print(banner)

    def print_table(df, super_hdrs, col_config):
        """Print a table using the super-headers to determine column order."""
        if df.empty:
            print("* No variables")
            return

        # Build the final columns list from super_headers
        columns_list = get_ordered_columns(super_hdrs, col_config)

        # 1) Super-header line
        super_line_parts = []
        for block in super_hdrs:
            block_width = sum(col_config[col_key]["width"] for col_key in block["columns"])
            if block["align"] == "center":
                super_line_parts.append(f"{block['title']:^{block_width}}")
            elif block["align"] == "right":
                super_line_parts.append(f"{block['title']:>{block_width}}")
            else:  # left
                super_line_parts.append(f"{block['title']:<{block_width}}")
        super_line = "".join(super_line_parts)

        # 2) Column header line
        header_line = "".join(
            f"{c['header']:<{c['width']}}" for c in columns_list
        )

        # 3) Dash line
        total_width = sum(c["width"] for c in columns_list)
        dash_line = "-" * total_width

        print(super_line)
        print(header_line)
        print(dash_line)

        # 4) Data rows
        for _, row in df.iterrows():
            row_str_parts = []
            for c in columns_list:
                key = c["key"]
                val = row[key]
                if key in ("coverage_ts", "coverage_os"):
                    cell = format_number(val, coverage=True)
                elif key in ("median", "prob_reject", "p_value", "p_value_lower", "p_value_upper", "lower_os", "lower_os_1", "lower_os_2", "lower_ts", "upper_ts"):
                    cell = format_number(val)
                else:
                    cell = str(val)
                row_str_parts.append(f"{cell:<{c['width']}}")
            print("".join(row_str_parts))

    # ------------------------ 4) Main Logic ------------------------
    # 4.1 Print the high-level info in two columns
    sig_cols = get_ordered_columns(super_headers_significant, columns_config)
    print_info(info_lines, sig_cols)

    # Convert 'type' to short forms
    type_map = {'continuous': 'con', 'discrete': 'dis', 'categorical': 'cat'}
    result['type'] = result['type'].map(type_map).fillna(result['type'])
    result['rank'] = result['rank'].fillna(-1).astype(int)

    # Separate data
    sig_df = result[result['significance'] == 1].sort_values("rank")

    incon_df = (
        result[(result['significance'].isna()) & (result['type'] != 'ignored')]
        .assign(rank="-")
        .sort_values('variable_index')
    )

    insig_df = (
        result[result['significance'] == 0]
        .assign(rank="-")
        .sort_values('variable_index')
    )

    ign_df = result[result['type'] == 'ignored']

    # 4.2 Print significant table
    if sig_df.empty:
        print("[Significant Variables]")
        print("* No significant variables")
    else:
        print_table(sig_df, super_headers_significant, columns_config)
    print(make_banner_line(sig_cols))  # dynamic '=' line

    # 4.3 Print inconclusive table
    if seed is None:
        incon_cols = get_ordered_columns(super_headers_inconclusive, columns_config)
        if incon_df.empty:
            print("[Inconclusive Variables]")
            print("* No inconclusive variables")
        else:
            print_table(incon_df, super_headers_inconclusive, columns_config)
        print(make_banner_line(incon_cols))

    # 4.4 Print insignificant table
    insig_cols = get_ordered_columns(super_headers_insignificant, columns_config)
    if insig_df.empty:
        print("[Insignificant Variables]")
        print("* No insignificant variables")
    else:
        print_table(insig_df, super_headers_insignificant, columns_config)
    print(make_banner_line(insig_cols))

    # 4.5 Ignored variables
    print("[Ignored Variables]")
    if ign_df.empty:
        print("* No ignored variables")
    else:
        for v in ign_df['variable']:
            print(f" - {v}")
    print(make_banner_line(insig_cols))
