import pytest
import pandas as pd
from imputegaps.impute_gaps import ImputeGaps

__author__ = "EMSK"
__copyright__ = "EMSK"
__license__ = "MIT"

IMPUTATION_METHODS_pick1_mean = {
    "pick1": ["dict", "bool"],
    "pick": None,
    "mode": None,
    "median": None,
    "skip": None,
    "mean": ["float", "percentage"],
    "nan": None,
}

IMPUTATION_METHODS_nan_mean = {
    "pick1": ["int", "index", "str", "date", "undefined"],
    "pick": None,
    "mode": None,
    "median": None,
    "skip": None,
    "mean": ["float", "percentage"],
    "nan": ["dict", "bool", "int"],
}

IMPUTATION_METHODS_pick_median = {
    "pick": ["dict", "bool", "int", "index", "str", "date", "undefined"],
    "pick1": None,
    "mode": None,
    "median": ["float", "percentage"],
    "skip": None,
    "mean": None,
    "nan": ["int"],
}

IMPUTATION_METHODS_pick_mode = {
    "pick": ["dict", "bool", "int", "index", "str", "date", "undefined"],
    "pick1": None,
    "mode": ["float", "percentage"],
    "median": None,
    "skip": None,
    "mean": None,
    "nan": ["int"],
}


ID_KEY = "be_id"
SET_SEED = 1
GROUP_BY = {"gk6sbi2": {"dimensions": ["gk", "sbi"]}, "drop_dimensions": True}


def test_pick_met_bool():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1],
            [2, 1, "A", "10", 1],
            [3, 1, "A", "10", 1],
            [4, 1, "A", "10", 1],
            [5, 1, "B", "10", 0],
            [6, 1, "B", "10", 0],
            [7, 1, "B", "10", 0],
            [8, 1, "B", "10", 0],
            [
                9,
                1,
                "B",
                "10",
                None,
            ],  # imputatie in eerste ronde, geen valid donor in tweede ronde
            [
                10,
                1,
                "B",
                "10",
                None,
            ],  # imputatie in eerste ronde, geen valid donor in tweede ronde
            [11, 1, "C", "10", None],  # imputatie in tweede ronde
            [12, 1, "C", "10", None],  # imputatie in tweede ronde
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "bool",
            "no_impute": False,
            "filter": "internet",
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_pick_median,
        track_imputed=True,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [float(1), 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1], copy=False, name="telewerkers"
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_pick1_met_bool():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1],
            [2, 1, "A", "10", 1],
            [3, 1, "A", "10", 1],
            [4, 1, "A", "10", 1],
            [5, 1, "B", "10", 0],
            [6, 1, "B", "10", 0],
            [7, 1, "B", "10", 0],
            [8, 1, "B", "10", 0],
            [9, 1, "B", "10", None],
            [10, 1, "B", "10", None],
            [11, 1, "C", "10", None],
            [12, 1, "C", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "bool",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_pick1_mean,
        track_imputed=True,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [float(1), 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1], copy=False, name="telewerkers"
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_nan_met_bool():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1],
            [2, 1, "A", "10", 1],
            [3, 1, "A", "10", 1],
            [4, 1, "A", "10", 1],
            [5, 1, "B", "10", 0],
            [6, 1, "B", "10", 0],
            [7, 1, "B", "10", 0],
            [8, 1, "B", "10", 0],
            [9, 1, "B", "10", None],
            [10, 1, "B", "10", None],
            [11, 1, "C", "10", None],
            [12, 1, "C", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "bool",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_nan_mean,
        track_imputed=True,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [float(1), 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], copy=False, name="telewerkers"
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_exclude_invalid_mean_met_float():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 30],
            [4, 1, "A", "10", 40],
            [5, 1, "B", "10", 50],
            [6, 1, "B", "10", 60],
            [7, 1, "B", "10", 70],
            [8, 1, "B", "10", 80],
            [9, 1, "B", "10", 85],
            [
                10,
                1,
                "B",
                "10",
                None,
            ],  # imputatie in eerste ronde, geen valid donor in tweede ronde
            [
                11,
                1,
                "B",
                "10",
                None,
            ],  # imputatie in eerste ronde, geen valid donor in tweede ronde
            [12, 1, "C", "10", None],  # imputatie in tweede ronde
            [13, 1, "C", "10", None],  # imputatie in tweede ronde
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_nan_mean,
        track_imputed=True,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [
            float(10),
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            85,
            69,
            69,
            49.44444444444444,
            49.44444444444444,
        ],
        copy=False,
        name="telewerkers",
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_exclude_invalid_median_met_float():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 30],
            [4, 1, "A", "10", 40],
            [5, 1, "B", "10", 50],
            [6, 1, "B", "10", 60],
            [7, 1, "B", "10", 70],
            [8, 1, "B", "10", 80],
            [9, 1, "B", "10", 85],
            [
                10,
                1,
                "B",
                "10",
                None,
            ],  # imputatie in eerste ronde, geen valid donor in tweede ronde
            [
                11,
                1,
                "B",
                "10",
                None,
            ],  # imputatie in eerste ronde, geen valid donor in tweede ronde
            [12, 1, "C", "10", None],  # imputatie in tweede ronde
            [13, 1, "C", "10", None],  # imputatie in tweede ronde
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_pick_median,
        track_imputed=True,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [float(10), 20, 30, 40, 50, 60, 70, 80, 85, 70, 70, 50, 50],
        copy=False,
        name="telewerkers",
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_exclude_invalid_modus_met_float():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 30],
            [4, 1, "A", "10", 40],
            [5, 1, "B", "10", 50],
            [6, 1, "B", "10", 60],
            [7, 1, "B", "10", 70],
            [8, 1, "B", "10", 80],
            [9, 1, "B", "10", 85],
            [
                10,
                1,
                "B",
                "10",
                None,
            ],  # imputatie in eerste ronde, geen valid donor in tweede ronde
            [
                11,
                1,
                "B",
                "10",
                None,
            ],  # imputatie in eerste ronde, geen valid donor in tweede ronde
            [12, 1, "C", "10", None],  # imputatie in tweede ronde
            [13, 1, "C", "10", None],  # imputatie in tweede ronde
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_pick_mode,
        track_imputed=True,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [float(10), 20, 30, 40, 50, 60, 70, 80, 85, 50, 50, 10, 10],
        copy=False,
        name="telewerkers",
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_voor_filter():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 1],
            [2, 1, "A", "10", 1],
            [3, 2, "A", "10", 1],
            [4, 2, "A", "10", 1],
            [5, 1, "B", "10", 0],
            [6, 1, "B", "10", 0],
            [7, 1, "B", "10", 0],
            [8, 1, "B", "10", 0],
            [9, 2, "B", "10", None],
            [10, 1, "B", "10", None],
            [11, 2, "C", "10", None],
            [12, 2, "C", "10", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "bool",
            "no_impute": False,
            "filter": "internet",
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_pick_mode,
        track_imputed=True,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [float(1), 1, 1, 1, 0, 0, 0, 0, None, 0, None, None],
        copy=False,
        name="telewerkers",
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_exclude_invalid_derde_ronde():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 30],
            [4, 1, "A", "10", 40],
            [5, 1, "B", "10", 50],
            [6, 1, "B", "10", 60],
            [7, 1, "B", "10", 70],
            [8, 1, "B", "10", 80],
            [9, 1, "B", "10", 85],
            [
                10,
                1,
                "B",
                "10",
                None,
            ],  # imputatie in eerste ronde, geen valid donor in tweede ronde
            [
                11,
                1,
                "B",
                "10",
                None,
            ],  # imputatie in eerste ronde, geen valid donor in tweede ronde
            [12, 1, "C", "10", None],  # imputatie in tweede ronde
            [13, 1, "C", "10", None],  # imputatie in tweede ronde
            [14, 1, "D", "20", None],  # imputatie in derde ronde
            [15, 1, "D", "20", None],  # imputatie in derde ronde
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_nan_mean,
        track_imputed=True,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [
            float(10),
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            85,
            69,
            69,
            49.44444444444444,
            49.44444444444444,
            49.44444444444444,
            49.44444444444444,
        ],
        copy=False,
        name="telewerkers",
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_exclude_invalid_hele_dataset():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", None],
            [9, 1, "B", "10", None],
            [13, 1, "C", "10", None],
            [15, 1, "D", "20", None],
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_nan_mean,
        track_imputed=True,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series([None, None, None, None], copy=False, name="telewerkers")

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_exclude_invalid_min_threshold_n5():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 30],
            [4, 1, "A", "10", 40],
            [
                12,
                1,
                "A",
                "10",
                None,
            ],  # imputatie in A/10 lukt niet, dus o.b.v. 10 in tweede ronde
            [
                13,
                1,
                "A",
                "10",
                None,
            ],  # imputatie in A/10 lukt niet, dus o.b.v. 10 in tweede ronde
            [5, 1, "B", "10", 50],
            [6, 1, "B", "10", 60],
            [7, 1, "B", "10", 70],
            [8, 1, "B", "10", 80],
            [9, 1, "B", "10", 85],
            [
                10,
                1,
                "B",
                "10",
                None,
            ],  # imputatie o.b.v. B/10 lukt, geen valid donor in tweede ronde
            [
                11,
                1,
                "B",
                "10",
                None,
            ],  # imputatie o.b.v. B/10 lukt, geen valid donor in tweede ronde
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_nan_mean,
        track_imputed=True,
        min_threshold=5,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [
            10,
            20,
            30,
            40,
            49.44444444444444,
            49.44444444444444,
            50,
            60,
            70,
            80,
            85,
            69,
            69,
        ],
        copy=False,
        name="telewerkers",
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_exclude_invalid_min_threshold_n10():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 30],
            [4, 1, "A", "10", 40],
            [12, 1, "A", "10", None],  # imputatie lukt niet want n < 10
            [13, 1, "A", "10", None],  # imputatie lukt niet want n < 10
            [5, 1, "B", "10", 50],
            [6, 1, "B", "10", 60],
            [7, 1, "B", "10", 70],
            [8, 1, "B", "10", 80],
            [9, 1, "B", "10", 85],
            [10, 1, "B", "10", None],  # imputatie lukt niet want n < 10
            [11, 1, "B", "10", None],  # imputatie lukt niet want n < 10
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_nan_mean,
        track_imputed=True,
        min_threshold=10,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [10, 20, 30, 40, None, None, 50, 60, 70, 80, 85, None, None],
        copy=False,
        name="telewerkers",
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_track_imputed_false():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 1, "A", "10", 10],
            [2, 1, "A", "10", 20],
            [3, 1, "A", "10", 30],
            [4, 1, "A", "10", 40],
            [5, 1, "B", "10", 50],
            [6, 1, "B", "10", 60],
            [7, 1, "B", "10", 70],
            [8, 1, "B", "10", 80],
            [9, 1, "B", "10", 85],
            [
                10,
                1,
                "B",
                "10",
                None,
            ],
            # imputatie in eerste ronde, wel valid donor in tweede ronde
            [
                11,
                1,
                "B",
                "10",
                None,
            ],
            # imputatie in eerste ronde, wel valid donor in tweede ronde
            [12, 1, "C", "10", None],  # imputatie in tweede ronde
            [13, 1, "C", "10", None],  # imputatie in tweede ronde
        ],
        columns=["be_id", "internet", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_nan_mean,
        track_imputed=False,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [float(10), 20, 30, 40, 50, 60, 70, 80, 85, 69, 69, 53, 53],
        copy=False,
        name="telewerkers",
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


def test_set_nan_eval():
    """
    var_type: float
    missing: 1
    imputatie: o.b.v. sbi x gk
    methode: median
    """
    # Maak data

    records_df = pd.DataFrame(
        [
            [1, 21, "A", "10", 10],
            [2, 21, "A", "10", 20],
            [3, 21, "A", "10", 30],
            [4, 21, "A", "10", 40],
            [5, 21, "B", "10", 50],
            [6, 21, "B", "10", 60],
            [7, 21, "B", "10", 70],
            [8, 21, "B", "10", 80],
            [9, 21, "B", "10", 85],
            [10, 10, "B", "10", None],
            [11, 10, "B", "10", None],
            [12, 10, "C", "10", None],
            [13, 10, "C", "10", None],
        ],
        columns=["be_id", "gk_sbs", "sbi", "gk", "telewerkers"],
    )
    variables = {
        "telewerkers": {
            "type": "float",
            "no_impute": False,
            "filter": None,
            "impute_only": None,
            "set_nan_eval": "gk_sbs < 20"
        }
    }

    # Run ImputeGaps
    impute_gaps = ImputeGaps(
        variables=variables,
        imputation_methods=IMPUTATION_METHODS_nan_mean,
        track_imputed=False,
        min_threshold=2,
        index_key=ID_KEY,
        seed=SET_SEED,
    )
    new_records = impute_gaps.impute_gaps(
        records_df=records_df, group_by=["gk", "sbi"], drop_dimensions=True
    )

    # Maak verwacht
    expected = pd.Series(
        [float(10), 20, 30, 40, 50, 60, 70, 80, 85, None, None, None, None],
        copy=False,
        name="telewerkers",
    )

    # Test uitvoeren
    pd.testing.assert_series_equal(new_records["telewerkers"], expected)


# TODO: Dropdimensions testen
