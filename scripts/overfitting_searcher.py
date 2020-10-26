"""
A script for detecting where overfitting occurs, based on metadata of a EDNEL run.
"""

import argparse


def main(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='For detecting how many runs had overfitting.'
    )

    parser.add_argument(
        '--experiment-path', action='store', required=False, default=None,
        help='Path to folder with experiment metadata.'
    )

    main(parser.parse_args())
