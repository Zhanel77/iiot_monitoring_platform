import math


def adapt_raw_payload(raw: dict) -> dict:
    air_temp = raw["Air temperature [K]"]
    process_temp = raw["Process temperature [K]"]
    rpm = raw["Rotational speed [rpm]"]
    torque = raw["Torque [Nm]"]
    tool_wear = raw["Tool wear [min]"]

    temp_diff = process_temp - air_temp
    power_kw = 2 * math.pi * torque * rpm / 60.0 / 1000.0

    return {
        "Air temperature [K]": air_temp,
        "temp_diff": temp_diff,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "power_kw": power_kw,
        "Tool wear [min]": tool_wear,
    }