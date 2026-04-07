CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    device_id VARCHAR(100) NOT NULL,
    machine_id INTEGER NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    source VARCHAR(50),
    scenario VARCHAR(50),
    prediction INTEGER NOT NULL,
    risk_score DOUBLE PRECISION,
    risk_level VARCHAR(20),
    model_type VARCHAR(50) DEFAULT 'edge',
    created_at TIMESTAMPTZ DEFAULT NOW()
);