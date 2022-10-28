#!/usr/bin/python3
import argparse

from cairo_planning.geometric.transformation import quat2rpy

def main():
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__)
    required = parser.add_argument_group('required arguments')

    parser.add_argument(
        '-o', '--orientation', dest='orientation', nargs=4, type=float, required=False,
        help='the w, x, y, z quaternion orientation of the end-effector'
    )

    args = parser.parse_args()

    if args.orientation is None:
        orientation = [1, 0, 0, 0]
    else:
        orientation = args.orientation

    print(list(quat2rpy(orientation)))

if __name__ == "__main__":
    main()

