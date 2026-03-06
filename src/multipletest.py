import numpy as np
import pandas as pd


def _extract_p_values(result: pd.DataFrame, p_col: str) -> pd.Series:
    if p_col in result and result[p_col].notna().any():
        p = result[p_col].copy()
    elif {"p_value_lower", "p_value_upper"}.issubset(result.columns):
        p = (result["p_value_lower"] + result["p_value_upper"]) / 2.0
    else:
        raise ValueError("Result must contain either p_value or p_value_lower/p_value_upper.")

    if "type" in result.columns:
        mask = result["type"] != "ignored"
    else:
        mask = pd.Series(True, index=result.index)

    p = p.where(mask)
    return p


def apply_fdr(result: pd.DataFrame, alpha: float = 0.05, method: str = "BH", p_col: str = "p_value") -> pd.DataFrame:
    """
    Apply FDR control to AICO results.

    Supported methods:
      - 'BH': Benjamini–Hochberg (independence / PRDS).
      - 'BY': Benjamini–Yekutieli (arbitrary dependence, more conservative).
    """
    if method not in {"BH", "BY"}:
        raise ValueError(f"Unknown FDR method '{method}'.")

    p = _extract_p_values(result, p_col)
    valid = p.notna()
    p_valid = p[valid].to_numpy()

    m = p_valid.size
    if m == 0:
        result["q_value"] = np.nan
        result["significant_fdr"] = False
        return result

    order = np.argsort(p_valid)
    ranks = np.arange(1, m + 1, dtype=float)
    p_sorted = p_valid[order]

    if method == "BH":
        factor = 1.0
    else:
        factor = np.sum(1.0 / ranks)

    q_sorted = factor * m * p_sorted / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(q_sorted)
    q[order] = np.minimum(q_sorted, 1.0)

    q_full = pd.Series(np.nan, index=result.index)
    q_full.loc[valid.index[valid]] = q

    result = result.copy()
    result["q_value"] = q_full
    result["significant_fdr"] = result["q_value"] <= alpha
    result.loc[~valid, ["q_value", "significant_fdr"]] = [np.nan, False]
    return result


def apply_fwer(result: pd.DataFrame, alpha: float = 0.05, method: str = "holm", p_col: str = "p_value") -> pd.DataFrame:
    """
    Apply FWER control to AICO results.

    Supported methods:
      - 'holm': Holm–Bonferroni step-down.
      - 'bonferroni': classic Bonferroni adjustment.
    """
    if method not in {"holm", "bonferroni"}:
        raise ValueError(f"Unknown FWER method '{method}'.")

    p = _extract_p_values(result, p_col)
    valid = p.notna()
    p_valid = p[valid].to_numpy()

    m = p_valid.size
    if m == 0:
        result["p_value_fwer"] = np.nan
        result["significant_fwer"] = False
        return result

    result = result.copy()

    if method == "bonferroni":
        p_adj = np.minimum(p_valid * m, 1.0)
    else:
        order = np.argsort(p_valid)
        p_sorted = p_valid[order]
        ranks = np.arange(1, m + 1, dtype=float)
        p_holm = (m - ranks + 1) * p_sorted
        p_holm = np.maximum.accumulate(p_holm)
        p_holm = np.minimum(p_holm, 1.0)
        p_adj = np.empty_like(p_holm)
        p_adj[order] = p_holm

    p_adj_full = pd.Series(np.nan, index=result.index)
    p_adj_full.loc[valid.index[valid]] = p_adj

    result["p_value_fwer"] = p_adj_full
    result["significant_fwer"] = result["p_value_fwer"] <= alpha
    result.loc[~valid, ["p_value_fwer", "significant_fwer"]] = [np.nan, False]
    return result

