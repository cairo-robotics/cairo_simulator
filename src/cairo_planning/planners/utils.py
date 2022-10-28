from decimal import Decimal, localcontext, ROUND_DOWN

import numpy as np

def name2idx(igraph, name):
    try:
        return igraph.vs.find(name).index
    except Exception as e:
        return None

def val2str(value, decimal_places=8):
    def trunc(number, places=decimal_places):
        if not isinstance(places, int):
            raise ValueError("Decimal places must be an integer.")
        if places < 1:
            raise ValueError("Decimal places must be at least 1.")
        # If you want to truncate to 0 decimal places, just do int(number).

        with localcontext() as context:
            context.rounding = ROUND_DOWN
            exponent = Decimal(str(10 ** - places))
            return Decimal(str(number)).quantize(exponent).to_eng_string()
    return str([trunc(num, decimal_places) for num in value])


def val2idx(igraph, value):
    return name2idx(igraph, val2str(value))

