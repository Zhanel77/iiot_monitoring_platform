from copy import deepcopy


def apply_scenario(row: dict, scenario_name: str) -> dict:
    row_copy = deepcopy(row)

    if scenario_name == "normal_run":
        return row_copy

    if scenario_name == "high_temp":
        row_copy["Process temperature [K]"] += 3.0
        row_copy["Air temperature [K]"] += 1.0
        return row_copy

    if scenario_name == "high_torque":
        row_copy["Torque [Nm]"] += 5.0
        return row_copy

    return row_copy