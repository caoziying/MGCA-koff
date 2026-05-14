from __future__ import annotations

from utils.trainer import parse_args, train_cross_validation


def main():
    args = parse_args()
    train_cross_validation(args)


if __name__ == "__main__":
    main()
