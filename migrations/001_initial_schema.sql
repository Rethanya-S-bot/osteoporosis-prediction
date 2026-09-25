-- =============================================================================
-- 001_initial_schema.sql
-- Osteoporosis Prediction — Supabase (PostgreSQL) Migration
--
-- Apply via: Supabase Dashboard → SQL Editor → paste & run
-- Or: psql $DATABASE_URL -f migrations/001_initial_schema.sql
-- =============================================================================

-- Enable uuid extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- TABLE: datasets
-- Tracks which datasets have been loaded
-- =============================================================================
CREATE TABLE IF NOT EXISTS datasets (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT NOT NULL,
    source_url  TEXT,
    rows        INTEGER,
    features    INTEGER,
    loaded_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- TABLE: patients
-- Core patient demographic & clinical data
-- =============================================================================
CREATE TABLE IF NOT EXISTS patients (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    age                 INTEGER NOT NULL CHECK (age BETWEEN 1 AND 120),
    gender              TEXT NOT NULL CHECK (gender IN ('Male','Female','Other')),
    hormonal_changes    TEXT,
    family_history      TEXT,
    race_ethnicity      TEXT,
    body_weight_bmi     NUMERIC(5,2),
    calcium_intake      TEXT,
    vitamin_d_intake    TEXT,
    physical_activity   TEXT,
    smoking             TEXT,
    alcohol_consumption TEXT,
    medical_conditions  TEXT,
    medications         TEXT,
    prior_fractures     TEXT,
    bmd_t_score         NUMERIC(5,2),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- TABLE: prediction_history
-- Each prediction result linked to a patient record
-- =============================================================================
CREATE TABLE IF NOT EXISTS prediction_history (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID REFERENCES patients(id) ON DELETE SET NULL,
    model_name      TEXT NOT NULL,
    risk_label      TEXT NOT NULL CHECK (risk_label IN ('Normal','Osteopenia','Osteoporosis')),
    confidence      NUMERIC(5,4),          -- probability of predicted class
    probabilities   JSONB,                 -- {"Normal": 0.1, "Osteopenia": 0.3, "Osteoporosis": 0.6}
    input_data      JSONB,                 -- all submitted patient fields
    prediction_result TEXT,                -- explicit result alias for API consumers
    shap_values     JSONB,                 -- {"Age": 0.23, "BMD_T_Score": -0.51, ...}
    llm_summary     TEXT,                  -- plain-language narrative
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE prediction_history ADD COLUMN IF NOT EXISTS input_data JSONB;
ALTER TABLE prediction_history ADD COLUMN IF NOT EXISTS prediction_result TEXT;

-- =============================================================================
-- TABLE: model_runs
-- Training run metadata and evaluation metrics
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name      TEXT NOT NULL,
    accuracy        NUMERIC(6,4),
    precision_macro NUMERIC(6,4),
    recall_macro    NUMERIC(6,4),
    f1_macro        NUMERIC(6,4),
    roc_auc         NUMERIC(6,4),
    hyperparams     JSONB,
    cv_folds        INTEGER,
    train_rows      INTEGER,
    test_rows       INTEGER,
    is_best         BOOLEAN DEFAULT FALSE,
    trained_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- INDEXES
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_prediction_history_patient    ON prediction_history(patient_id);
CREATE INDEX IF NOT EXISTS idx_prediction_history_created_at ON prediction_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_history_risk_label ON prediction_history(risk_label);
CREATE INDEX IF NOT EXISTS idx_model_runs_trained_at        ON model_runs(trained_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_runs_is_best           ON model_runs(is_best) WHERE is_best = TRUE;

-- =============================================================================
-- ROW LEVEL SECURITY (RLS)
-- =============================================================================
ALTER TABLE datasets            ENABLE ROW LEVEL SECURITY;
ALTER TABLE patients            ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_history  ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_runs          ENABLE ROW LEVEL SECURITY;

-- Anon users: read-only access (for the Streamlit /history page)
CREATE POLICY "anon_select_datasets"
    ON datasets FOR SELECT USING (TRUE);

CREATE POLICY "anon_select_patients"
    ON patients FOR SELECT USING (TRUE);

CREATE POLICY "anon_insert_patients"
    ON patients FOR INSERT WITH CHECK (TRUE);

CREATE POLICY "anon_select_prediction_history"
    ON prediction_history FOR SELECT USING (TRUE);

CREATE POLICY "anon_insert_prediction_history"
    ON prediction_history FOR INSERT WITH CHECK (TRUE);

CREATE POLICY "anon_select_model_runs"
    ON model_runs FOR SELECT USING (TRUE);

-- Service role (your backend): full access (uses service_role key, not anon)
CREATE POLICY "service_all_datasets"
    ON datasets FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "service_all_patients"
    ON patients FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "service_all_prediction_history"
    ON prediction_history FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "service_all_model_runs"
    ON model_runs FOR ALL USING (auth.role() = 'service_role');
