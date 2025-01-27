import pandas as pd

from imputegaps.impute_gaps import ImputeGaps

__author__ = "EMSK"
__copyright__ = "EMSK"
__license__ = "MIT"

IMPUTATION_METHODS = {"pick": ["float", "percentage", "dict", "bool", "int"]}
ID_KEY = "be_id"
SET_SEED = 2
GROUP_BY = {"gk6sbi2": {"dimensions": ["gk", "sbi"]}, "drop_dimensions": True}

# Dit script bevat de volgende tests:
# - Tests voor de volgende var_types: 'float', 'percentage', 'bool', 'dict'.
#   Al deze testen bevatten een situatie waarin er één missende waarde, waarbij:
#       * Eentje leeg in stratum ['sbi', 'gk'] -> imputeren o.b.v. SBI en GK
#       * Alles leeg in stratum ['sbi', 'gk'], maar niet in ['gk'] -> imputeren o.b.v GK
#       * Alles leeg in stratum ['sbi', 'gk'], maar ook in ['gk'] -> imputeren o.b.v. hele dataset
# - Test voor een situatie met een filter.


def test_float():
    """
    Test voor var_type 'float' met imputatiemethode 'pick'
    """
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1.1],
            [2, 1, "A", "10", 2.2],
            [3, 2, "A", "10", 3.3],
            [4, 2, "A", "10", 4.4],
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

    new_records = impute_gaps.impute_gaps(records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True)

    # Expected
    expected = pd.Series([1.1, 2.2, 3.3, 4.4, 1.1, 1.1, 4.4], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_percentage():
    """
    Test voor var_type 'percentage' met imputatiemethode 'pick'
    """
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1.1],
            [2, 1, "A", "10", 2.2],
            [3, 2, "A", "10", 3.3],
            [4, 2, "A", "10", 4.4],
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
            "type": "percentage",
        }
    }

    # Init ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS,
        index_key=ID_KEY,
        seed=SET_SEED,
    )

    new_records = impute_gaps.impute_gaps(records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True)

    # Expected
    expected = pd.Series([1.1, 2.2, 3.3, 4.4, 1.1, 1.1, 4.4], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_bool():
    """
    Test voor var_type 'bool' met imputatiemethode 'pick'
    """
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1],
            [2, 1, "A", "10", 0],
            [3, 2, "A", "10", 1],
            [4, 2, "A", "10", 0],
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
            "type": "bool",
        }
    }

    # Init ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS,
        index_key=ID_KEY,
        seed=SET_SEED,
    )

    new_records = impute_gaps.impute_gaps(records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True)

    # Expected
    expected = pd.Series([float(1), 0, 1, 0, 1, 1, 0], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_dict():
    """
    Test for var_type 'bool' with imputation method 'pick'
    """
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1],
            [2, 1, "A", "10", 2],
            [3, 2, "A", "10", 3],
            [4, 2, "A", "10", 4],
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
            "type": "dict",
        }
    }

    # Init ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS,
        index_key=ID_KEY,
        seed=SET_SEED,
    )

    new_records = impute_gaps.impute_gaps(records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True)

    # Expected
    # The expected values are floats as the initial values are floats due to the Nones
    expected_telewerkers = pd.Series([1.0, 2, 3, 4, 1, 1, 4], copy=False, name="telewerkers")
    new_telewerkers = new_records["telewerkers"]

    # Test uitvoeren
    pd.testing.assert_series_equal(new_telewerkers, expected_telewerkers)


def test_met_filter():
    """
    Test voor var_type 'dict' met imputatiemethode 'pick'
    """
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1],
            [2, 1, "A", "10", 2],
            [3, 2, "A", "10", 3],
            [4, 2, "A", "10", 4],
            # Valt buiten het filter dus wordt niet geïmputeerd
            [5, 2, "A", "10", None],
            [6, 1, "C", "10", None],
            [7, 1, "C", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {"telewerkers": {"type": "dict", "filter": "internet"}}

    # Init ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS,
        index_key=ID_KEY,
        seed=SET_SEED,
    )

    new_records = impute_gaps.impute_gaps(records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True)

    # Expected
    expected = pd.Series([float(1), 2, 3, 4, None, 1, 2], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)
