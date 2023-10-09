TOP_LE = {
    "name": "TOP",
    "unit_weight": 22,
    "YOUNG_MODULUS": 125e6,
    "POISSON_RATIO": 0.35,
}

MIDDLE_LE = {
    "name": "MIDDLE",
    "unit_weight": 15,
    "YOUNG_MODULUS": 6500000,
    "POISSON_RATIO": 0.35,
}

BOTTOM_LE = {
    "name": "BOTTOM",
    "unit_weight": 18,
    "YOUNG_MODULUS": 50.00E6,
    "POISSON_RATIO": 0.35,
}



TOP_MC = {
    "name": "TOP",
    "unit_weight": 22,
    "UMAT_PARAMETERS": {
        "UMAT_NAME": "umat_sand",
        "YOUNG_MODULUS": 125e6,
        "FRICTION_ANGLE": 39.8,
        "DILATANCY_ANGLE": 0.0,
        "COHESION": 1.0,
        "CUTOFF_STRENGTH": 0.0,
        "POISSON_RATIO": 0.35,
        "UNDRAINED_POISSON_RATIO": 0.35,
        "YIELD_FUNCTION_TYPE": 1,

    }
}

MIDDLE_MC = {
    "name": "MIDDLE",
    "unit_weight": 15,
    "UMAT_PARAMETERS": {
        "UMAT_NAME": "umat_sand",
        "YOUNG_MODULUS": 6500000,
        "FRICTION_ANGLE": 25.80,
        "DILATANCY_ANGLE": 0.0,
        "COHESION": 14.80,
        "CUTOFF_STRENGTH": 0.0,
        "POISSON_RATIO": 0.35,
        "UNDRAINED_POISSON_RATIO": 0.35,
        "YIELD_FUNCTION_TYPE": 1,

    }
}

BOTTOM_MC = {
    "name": "BOTTOM",
    "unit_weight": 18,
    "UMAT_PARAMETERS": {
        "UMAT_NAME": "umat_sand",
        "YOUNG_MODULUS": 50.00E6,
        "FRICTION_ANGLE": 37,
        "DILATANCY_ANGLE": 0.0,
        "COHESION": 1.0,
        "CUTOFF_STRENGTH": 0.0,
        "POISSON_RATIO": 0.35,
        "UNDRAINED_POISSON_RATIO": 0.35,
        "YIELD_FUNCTION_TYPE": 1,

    }
}