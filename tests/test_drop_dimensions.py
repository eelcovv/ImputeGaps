import pandas as pd

from imputegaps.impute_gaps import ImputeGaps

__author__ = "EMSK"
__copyright__ = "EMSK"
__license__ = "MIT"

IMPUTATION_METHODS = {"mean": ["float", "percentage", "dict", "bool", "int"]}
ID_KEY = "be_id"
SET_SEED = 2
GROUP_BY = {"gk6sbi2": {"dimensions": ["gk", "sbi"]}, "drop_dimensions": True}

# Dit script bevat de volgende tests:
# - Tests voor de volgende var_types: 'float', 'percentage'.
#   Al deze testen bevatten een situatie waarin er één missende waarde, waarbij:
#       * Eentje leeg in stratum ['sbi', 'gk'] -> imputeren o.b.v. SBI en GK
#       * Alles leeg in stratum ['sbi', 'gk'], maar niet in ['gk'] -> imputeren o.b.v GK
#       * Alles leeg in stratum ['sbi', 'gk'], maar ook in ['gk'] -> imputeren o.b.v. hele dataset
# - Test voor een situatie met een filter.


def test_drop_dimensions_no_drop():
    """
    Test voor var_type 'float' met imputatiemethode 'mean'
    """
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1.2],
            [2, 1, "A", "10", 2.3],
            [3, 2, "A", "10", 3.4],
            [4, 2, "A", "10", 4.5],
            # Eentje leeg in stratum ['sbi', 'gk']:
            [5, 2, "A", "10", None],
            # Alles leeg in stratum ['sbi', 'gk'], maar niet in ['gk']:
            [6, 1, "C", "10", None],
            # Alles leeg in stratum ['sbi', 'gk'], maar ook in ['gk']:
            [7, 1, "C", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
        }
    }

    # Init ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS,
        index_key=ID_KEY,
        seed=SET_SEED,
    )

    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=False
    )

    # Expected
    expected_telewerkers = pd.Series(
        [1.2, 2.3, 3.4, 4.5, 2.85, None, None], copy=False, name="telewerkers"
    )
    new_expected_telewerkers = new_records["telewerkers"]

    # Test uitvoeren
    pd.testing.assert_series_equal(new_expected_telewerkers, expected_telewerkers)


def test_drop_dimensions_drop():
    """
    Test voor var_type 'float' met imputatiemethode 'mean'
    """
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1.2],
            [2, 1, "A", "10", 2.3],
            [3, 2, "A", "10", 3.4],
            [4, 2, "A", "10", 4.5],
            # Eentje leeg in stratum ['sbi', 'gk']:
            [5, 2, "A", "10", None],
            # Alles leeg in stratum ['sbi', 'gk'], maar niet in ['gk']:
            [6, 1, "C", "10", None],
            # Alles leeg in stratum ['sbi', 'gk'], maar ook in ['gk']:
            [7, 1, "C", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
        }
    }

    # Init ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS,
        index_key=ID_KEY,
        seed=SET_SEED,
    )

    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Expected
    expected_telewerkers = pd.Series(
        [1.2, 2.3, 3.4, 4.5, 2.85, 2.85, 2.85], copy=False, name="telewerkers"
    )
    new_expected_telewerkers = new_records["telewerkers"]

    # Test uitvoeren
    pd.testing.assert_series_equal(new_expected_telewerkers, expected_telewerkers)
