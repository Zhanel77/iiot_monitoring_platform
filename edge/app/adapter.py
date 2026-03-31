from app.schemas import RawMQTTMessage, AdaptedFeatures


def adapt_raw_message(raw: RawMQTTMessage) -> AdaptedFeatures:
    sensors = raw.sensors

    features = {
        "Air temperature [K]": sensors.air_temperature_k,
        "Process temperature [K]": sensors.process_temperature_k,
        "Rotational speed [rpm]": sensors.rotational_speed_rpm,
        "Torque [Nm]": sensors.torque_nm,
        "Tool wear [min]": sensors.tool_wear_min,
    }

    return AdaptedFeatures(
        device_id=raw.device_id,
        machine_id=raw.machine_id,
        timestamp=raw.timestamp,
        source=raw.source,
        scenario=raw.scenario,
        features=features,
        ground_truth=raw.ground_truth.model_dump() if raw.ground_truth else None,
    )