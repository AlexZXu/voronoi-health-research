def load_ratio_color(value: float, average: float) -> str:
    if average <= 0:
        return "#dfe7ed"

    ratio = value / average
    if ratio < 0.7:
        return "#159447"
    if ratio < 0.9:
        return "#72c184"
    if ratio < 1.1:
        return "#b9de74"
    if ratio < 1.3:
        return "#f2d14b"
    if ratio < 1.5:
        return "#f59e42"
    return "#d64b3a"
