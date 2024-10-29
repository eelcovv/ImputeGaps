import pandas as pd

from imputegaps.impute_gaps import ImputeGaps

__author__ = "EMSK"
__copyright__ = "EMSK"
__license__ = "MIT"

# Dictionary containing imputation methods and their applicable variable types
IMPUTATION_METHODS = {"mean": ["float", "percentage", "dict", "bool", "int"]}
# Key used for indexing
ID_KEY = "be_id"
# Seed value for reproducibility
SET_SEED = 2
# Dictionary specifying the group by dimensions and whether to drop dimensions
GROUP_BY = {"gk6sbi2": {"dimensions": ["gk", "sbi"]}, "drop_dimensions": True}


# This script contains the following tests:
# - Tests for the following var_types: 'float', 'percentage'.
#   All these tests contain a situation with one missing value, where:
#       * One missing in stratum ['sbi', 'gk'] -> impute based on SBI and GK
#       * All missing in stratum ['sbi', 'gk'], but not in ['gk'] -> impute based on GK
#       * All missing in stratum ['sbi', 'gk'], and also in ['gk'] -> impute based on the entire
#         dataset
# - Test for a situation with a filter.


class TestDropDimensions:
    # DataFrame containing test records
    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1.2],
            [2, 1, "A", "10", 2.3],
            [3, 2, "A", "10", 3.4],
            [4, 2, "A", "10", 4.5],
            # One missing in stratum ['sbi', 'gk']:
            [5, 2, "A", "10", None],
            # All missing in stratum ['sbi', 'gk'], but not in ['gk']:
            [6, 1, "C", "10", None],
            # All missing in stratum ['sbi', 'gk'], and also in ['gk']:
            [7, 1, "C", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    # Dictionary specifying variable types
    variables = {
        "telewerkers": {
            "type": "float",
        }
    }

    def test_drop_dimensions_no_drop(self):
        """
        Test for var_type 'float' with imputation method 'mean' without dropping dimensions.
        """

        # Initialize ImputeGaps
        impute_gaps = ImputeGaps(
            variables=self.variables,
            imputation_methods=IMPUTATION_METHODS,
            index_key=ID_KEY,
            seed=SET_SEED,
        )

        # Perform imputation without dropping dimensions
        new_records = impute_gaps.impute_gaps(
            records_df=self.records_df, group_by=["gk", "sbi"], drop_dimensions=False
        )

        # Expected result
        expected_telewerkers = pd.Series(
            [1.2, 2.3, 3.4, 4.5, 2.85, None, None], copy=False, name="telewerkers"
        )
        new_expected_telewerkers = new_records["telewerkers"]

        # Execute test
        pd.testing.assert_series_equal(new_expected_telewerkers, expected_telewerkers)

    def test_drop_dimensions_drop(self):
        """
        Test for var_type 'float' with imputation method 'mean' with dropping dimensions.
        """
        # Initialize ImputeGaps
        impute_gaps = ImputeGaps(
            variables=self.variables,
            imputation_methods=IMPUTATION_METHODS,
            index_key=ID_KEY,
            seed=SET_SEED,
        )

        # Perform imputation with dropping dimensions
        new_records = impute_gaps.impute_gaps(
            records_df=self.records_df, group_by=["gk", "sbi"], drop_dimensions=True
        )

        # Expected result
        expected_telewerkers = pd.Series(
            [1.2, 2.3, 3.4, 4.5, 2.85, 2.85, 2.85], copy=False, name="telewerkers"
        )
        new_expected_telewerkers = new_records["telewerkers"]

        # Execute test
        pd.testing.assert_series_equal(new_expected_telewerkers, expected_telewerkers)
