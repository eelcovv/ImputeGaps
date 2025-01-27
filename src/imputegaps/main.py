import argparse
import codecs
import logging
import sys

import pandas as pd
import yaml

from imputegaps import __version__, logger
from imputegaps.impute_gaps import ImputeGaps


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as a list of strings
          (for example, ``["--help"]``).

    Returns:

      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("records_df", help="Name of the impute filename")
    parser.add_argument(
        "--output_filename",
        help="Name of the output filename. If not given, output is written to stdout",
    )
    parser.add_argument("--variables", help="Variables impute methods")
    parser.add_argument(
        "--impute_settings_file",
        help="Name of the settings file with the imputation method per type",
    )
    parser.add_argument("--group_by", help="Group by column name to impute")
    parser.add_argument("--id", help="Index column name of the smallest group")
    parser.add_argument(
        "--version",
        action="version",
        version=f"ImputeGaps version {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def main(args):
    """
    doc here
    """

    logger.debug("Starting class ImputeGaps.")

    # Get command line arguments and set up logging
    args = parse_args(args)
    logger.setLevel(args.loglevel)

    # Read input files
    records_df = pd.read_csv(args.records_df, sep=";")
    variables = pd.read_csv(args.variables, sep=";")
    index_key = args.id

    # Read the settings file
    with codecs.open(args.impute_settings, encoding="UTF-8") as stream:
        settings = yaml.load(stream=stream, Loader=yaml.Loader)

    impute_settings = settings["general"]["imputation"]

    # Convert variables to dictionary
    # variables.set_index("naam", inplace=True)
    variables = variables.to_dict("index")

    # Start class ImputeGaps
    impute_gaps = ImputeGaps(
        index_key=index_key,
        imputation_methods=impute_settings["imputation_methods"],
        seed=impute_settings["set_seed"],
        variables=variables,
    )

    records_df = impute_gaps.impute_gaps(records_df=records_df, group_by=args.group_by)

    logger.info("Class ImputeGaps has finished.")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as an entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
