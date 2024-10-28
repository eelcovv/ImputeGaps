import logging
import warnings
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DataFrameType = Union["DataFrame", None]
DataFrameLikeType = Union["DataFrame", "Series", None]
SeriesType = Union["Series"]


def fill_missing_data(
    stratum, invalid_donors, col_name, how, min_threshold, seed=None
) -> DataFrameLikeType:
    """Impute missing values for one variable of a particular stratum (subset)

    Parameters
    ----------
    stratum: pd.Series
        pd.Series with one column that contains missing values.
    how: String {"mean", "median", "pick", "nan", "pick1"},
        Method that should be used to fill the missing values;
        - mean: Impute with the mean
        - median: Impute with the median
        - pick: Impute with a random value (for categorical variables)
        - nan: Impute with the value 0
        - pick1: Impute with the value 1
    seed: int
        Seed needed for random generator. Will only by imposed for seed == 1

    Returns
    -------
    stratum_to_impute: DataFrameLikeType
        Series with imputed values

    """
    stratum_to_impute = stratum.copy()
    invalid_donors = invalid_donors

    # Create a mask with True for all missing values
    mask_is_na = stratum_to_impute.isnull()

    # Skip if there are no missing values
    if not mask_is_na.any():
        return stratum_to_impute

    # If applicable, only select valid donor records (i.e. if track records with imputed values)
    if invalid_donors is not None:
        overlap_index = invalid_donors.index.intersection(stratum_to_impute.index)
        mask_invalid_donors = invalid_donors.reindex(overlap_index)
        valid_donor_records = stratum_to_impute[~mask_is_na & ~mask_invalid_donors]
    else:
        valid_donor_records = stratum_to_impute[~mask_is_na]

    # If the number of valid donors is smaller than the min_threshold, imputation is not possible
    # This only applies to mean, mode and pick, because the other methods do not rely on donor records
    if valid_donor_records.size < min_threshold and how in ["mean", "mode", "pick"]:
        return stratum_to_impute

    # Impute depending on which method to use
    if how == "mean":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            imputed_values = np.full(
                mask_is_na.size, fill_value=valid_donor_records.mean()
            )
    elif how == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            imputed_values = np.full(
                mask_is_na.size, fill_value=valid_donor_records.median()
            )
    elif how == "mode":
        mode = valid_donor_records.mode()[0]
        imputed_values = np.full(mask_is_na.size, fill_value=mode)
    elif how == "nan":
        try:
            stratum_to_impute = stratum_to_impute.cat.add_categories(["0.0"])
        except AttributeError or ValueError:
            pass
        imputed_values = np.full(stratum_to_impute.isnull().sum(), fill_value="0.0")
    elif how == "pick1":
        try:
            stratum_to_impute = stratum_to_impute.cat.add_categories(["1.0"])
        except AttributeError:
            pass
        except ValueError:
            pass
        imputed_values = np.full(stratum_to_impute.isnull().sum(), fill_value="1.0")
    elif how == "pick":
        if seed == 1:
            # only for seed is 1 we imposed every time we enter a new cell. Generates less random results
            # but useful for reproduction of the data
            np.random.seed(seed)
        number_of_nans = mask_is_na.sum()
        imputed_values = np.random.choice(
            valid_donor_records.values, size=number_of_nans, replace=True
        )
    else:
        raise ValueError("Not a valid imputation method: {}.".format(how))

    # Fill the missing values with the values from imputed_values
    if imputed_values.size > 1:
        stratum_to_impute[mask_is_na] = imputed_values
    else:
        stratum_to_impute.loc[mask_is_na] = imputed_values
    return stratum_to_impute


class ImputeGaps:
    """
    Initializes the ImputeGaps object.

    Arguments
    ---------
    index_key: str
        Name of the variable by which a record is identified (e.g. be_id)
    variables: dict
        Dictionary with information about the variables to impute.
    imputation_methods: dict
        Dictionary with imputation methods per data type.
    seed: int
        Seed for random number generator. If seed == 1, seed will be imposed every time a cell is
        entered, forcing less 'random results'. For seed not equal to 1, the seed will be set only once.
        For seed is None, no seed will be imposed, meaning that your outcome of random pick will be differenct
        every time your run the code

    Notes
    ----------
    *   De dictionary 'variables' is in principe de pd.DataFrame 'self.variables' uit de ICT
        analyser, geconverteerd naar een dictionary.
        Als preprocessing stap moet hierbij de kolom 'filter' zijn platgeslagen, oftewel: het mag
        geen dictionary meer zijn. De dictionary 'variables' moet ten minste de volgende kolommen
        bevatten: [["type", "no_impute", "filter"]], met optioneel: "impute_only".
    *   De dict 'impute_settings' is een nieuw kopje onder 'General' in de settingsfile.
        Een belangrijk subkopje is 'group_by', wat er zo kan uitzien:
         group_by: "sbi_digit2, gk6_label; gk6_label"
        Dit betekent dat er eerst wordt geïmputeerd in strata o.b.v. sbi_digit2 en gk6_label.
        Als dat niet lukt, wordt alleen geïmputeerd o.b.v. gk6_label. Op dezelfde manier kunnen
        meer opties worden toegevoegd.
    """

    def __init__(
        self,
        index_key: str,
        imputation_methods: dict | None = None,
        variables: dict | None = None,
        seed: int = None,
        track_imputed: bool = False,
        min_threshold: int | None = 1,
    ):

        self.index_key = index_key
        self.imputation_methods = imputation_methods
        self.seed = seed
        self.track_imputed = track_imputed
        self.min_threshold = min_threshold
        self.variables = variables
        self.imputed_df = None

        if self.seed is not None:
            # Set seed for random number generator. Only needs to be done one time
            np.random.seed(seed)

    def impute_gaps(
        self,
        records_df: DataFrameType,
        group_by: list,
        drop_dimensions: bool = False,
    ) -> DataFrameType:
        """
        Impute all missing values in a dataframe for indices group_by.

        Parameters
        ----------
        records_df: DataFrameType
            DataFrame containing variables with missing values.
        group_by: list
            List with the variables by which the records should be grouped.
            The first variable is the most important one.
        drop_dimensions: bool



        Returns
        -------
        DataFrameType:
            DataFrame with imputed values.
        """

        original_indices = records_df.index.names
        records_df.reset_index(inplace=True)
        number_of_dimensions = len(group_by)

        # add one to number of dimensions because you add the index key
        for group_dim in range(number_of_dimensions + 1):
            max_dim = number_of_dimensions - group_dim
            if max_dim > 0:
                group_by_indices = group_by[:max_dim]
                index_for_group_by = [self.index_key] + group_by_indices
            else:
                index_for_group_by = self.index_key
                group_by_indices = []

            # set index before imputations
            records_df.set_index(index_for_group_by, inplace=True)

            if self.imputed_df is None and self.track_imputed:
                self.imputed_df = records_df.isna()
            elif self.imputed_df is not None and self.track_imputed:
                self.imputed_df = self.imputed_df.reset_index().set_index(
                    index_for_group_by
                )

            # Impute missing values for the new index
            records_df = self.impute_gaps_for_dimensions(
                records_df, group_by=group_by_indices
            )

            # after each iteration, reset the index
            records_df.reset_index(inplace=True)

            if not drop_dimensions:
                # by default, we do not continue imputing for the next group_by with one
                # less dimension
                break

        if drop_dimensions:
            # call the last time in case we gave drop dimensions
            records_df = self.impute_gaps_for_dimensions(records_df)

        if None not in original_indices:
            records_df.set_index(original_indices, inplace=True)
            logger.debug("Set index {}.".format(original_indices))

        return records_df

    def impute_gaps_for_dimensions(
        self, records_df: DataFrameType, group_by: list | None = None
    ) -> DataFrameType:
        """
        Impute all missing values in a dataframe for a particular subset (aka stratum).

        Parameters
        ----------
        records_df: DataFrameType
            DataFrame containing variables with missing values.
        group_by: list
            List with the new index for the DataFrame.

        Returns
        -------
        DataFrameType:
            DataFrame with imputed values for indices group_by.
        """

        # Set index to group_by
        how = None

        # Iterate over variables
        for col_name in records_df.columns:
            try:
                variable_properties = self.variables[col_name]
            except KeyError as err:
                logger.debug(f"Skip imputing, want geen variabele info voor {err}")
                continue
            # Check if information is available about the variable
            try:
                var_type = variable_properties["type"]
            except KeyError as err:
                logger.info(f"Geen 'type' info voor: {col_name}, {err}")
                continue

            no_impute = variable_properties.get("no_impute")
            skip_variable_type = self.imputation_methods.get("skip")

            # Check if the variable has a 'no_impute' flag or if its type should not be imputed
            if no_impute or (
                skip_variable_type is not None and var_type in skip_variable_type
            ):
                logger.debug(
                    "Skip imputing variable {} of var type {}".format(
                        col_name, var_type
                    )
                )
                continue

            # Get filter(s) if provided
            impute_only = variable_properties.get("impute_only")
            filter = variable_properties.get("filter")

            if (
                impute_only is None and filter is not None
            ):  # Als impute_only leeg is, neem dan filter
                var_filter = filter
            else:
                var_filter = impute_only

            # If a filter is provided, use it to filter the records
            mask_filter = pd.Series(True, index=records_df.index)
            if var_filter is not None:
                eval_str = var_filter + " == 1"
                try:
                    mask_filter = records_df.eval(eval_str, engine="python")
                except pd.errors.UndefinedVariableError as err:
                    logger.warning(
                        f"{err}\nImputation filter failed for {col_name} met {var_filter}"
                    )

            # If set_nan_eval is provided, use it to filter the records
            set_nan_eval = self.variables[col_name].get("set_nan_eval")
            mask_set_nan_eval = pd.Series(False, index=records_df.index)
            if set_nan_eval is not None:
                try:
                    mask_set_nan_eval = records_df.eval(set_nan_eval, engine="python")
                except pd.errors.UndefinedVariableError as err:
                    logger.warning(
                        f"{err}\nSet_nan_eval filter failed for {col_name} met {set_nan_eval}"
                    )

            col_to_impute = records_df[mask_filter & ~mask_set_nan_eval][col_name]

            if self.track_imputed:
                invalid_donors = self.imputed_df[col_name]
            else:
                invalid_donors = None

            start_type = col_to_impute.dtype

            # Compute number of missing values
            number_of_nans_before = col_to_impute.isnull().sum()
            column_size = col_to_impute.size

            # Skip if there are no missing values
            if number_of_nans_before == 0:
                logger.debug(f"Skip imputing {col_name}. It has no missing values.")
                continue

            # Skip if there is only missing values
            if number_of_nans_before == column_size:
                logger.debug(f"Skip imputing {col_name}. It has only missing values")
                continue

            logger.debug("Impute gaps {:20s} ({})".format(col_name, var_type))
            percentage_to_replace = round(100 * number_of_nans_before / column_size, 1)
            logger.debug(
                f"Filling {col_name} with {number_of_nans_before} / {column_size} nans "
                f"({percentage_to_replace:.1f} %)"
            )

            # Get which imputing method to use
            # TODO: Evt. veranderen en inbouwen dat je bijv. voor bools en dict geen mean en median kunt imputeren etc.
            imputation_dict = self.imputation_methods
            not_none = [
                i for i in imputation_dict.keys() if imputation_dict[i] is not None
            ]

            impute_method = variable_properties.get("impute_method")
            if impute_method is not None:
                how = impute_method
            else:
                for key, val in imputation_dict.items():
                    if key in not_none and var_type in imputation_dict[key]:
                        how = key
                        # Convert categorical (dict) variables to categorical
                        if var_type == "dict":
                            col_to_impute = col_to_impute.astype("category")
                        continue

            if how is None:
                logger.warning("Imputation method not found!")
            else:
                logger.debug(f"Fill gaps by taking the {how} of the valid values")

            def fill_gaps(stratum):
                """
                Impute missing values for one variable for a particular subset (aka stratum)

                Parameters
                ----------
                stratum: pd.Series
                    pd.Series with one column that contains missing values.

                Returns
                -------
                imputed_col: pd.Series
                    pd.Series with imputed values.
                """
                imputed_col = fill_missing_data(
                    stratum,
                    invalid_donors=invalid_donors,
                    col_name=col_name,
                    how=how,
                    min_threshold=self.min_threshold,
                    seed=self.seed,
                )
                return imputed_col

            # Iterate over the variables in the group_by-list and try to impute until there are no
            # more missing values
            if group_by:
                df_grouped = col_to_impute.groupby(
                    group_by, group_keys=False
                )  # Do group by
                col_to_impute = df_grouped.apply(fill_gaps)  # Impute missing values
            else:
                # for imputation on the whole column, we don't need to apply but just fill_gaps
                col_to_impute = fill_missing_data(
                    col_to_impute,
                    invalid_donors=invalid_donors,
                    how=how,
                    min_threshold=self.min_threshold,
                    col_name=col_name,
                    seed=self.seed,
                )

            number_of_nans_after = col_to_impute.isnull().sum()

            number_of_removed_nans = number_of_nans_before - number_of_nans_after

            if number_of_removed_nans == 0 and number_of_nans_before > 0:
                logger.warning(
                    f"Imputing {col_name} in stratum {group_by} - Failed imputing any gap: "
                    f"{number_of_removed_nans} gaps imputed / {number_of_nans_after} gaps remaining"
                )
            elif number_of_nans_after > 0:
                logger.warning(
                    f"Imputing {col_name} in stratum {group_by} - Failed imputing some gap: "
                    f"{number_of_removed_nans} gaps imputed / {number_of_nans_after} gaps remaining"
                )
            elif number_of_nans_after == 0:
                column_size = col_to_impute.size
                percentage_replaced = round(
                    100 * number_of_nans_before / column_size, 1
                )
                logger.info(
                    f"Imputing {col_name} in stratum {group_by} - Successfully imputed all {number_of_nans_before}/{column_size} "
                    f"({percentage_replaced:.1f} %) gaps."
                )
            else:
                logger.warning(
                    f"Imputing based on stratum {group_by} - Something went wrong with imputing gaps for {col_name}."
                )

            # Replace original column by imputed column
            records_df[col_name] = records_df[col_name].where(records_df[col_name].notna(), col_to_impute.astype(start_type))

        return records_df
