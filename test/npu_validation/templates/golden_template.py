#!/usr/bin/python3
# coding=utf-8

import numpy as np


def generate_golden(@GOLDEN_ARGS@):
    outputs = []
@GOLDEN_OUTPUTS@
    return @GOLDEN_RET@


def main():
    np.random.seed(19)
@INPUT_GENERATE@
    outputs = generate_golden(@GOLDEN_CALL_ARGS@)
@OUTPUT_WRITES@


if __name__ == "__main__":
    main()
