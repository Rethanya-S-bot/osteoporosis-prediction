from src.data_generator import generate_dataset
from src.inference import load_artifacts, predict_patient, save_artifacts
from src.training import train_and_compare


def test_training_comparison_returns_metrics_and_best_model():
    result = train_and_compare(generate_dataset(120, save=False))

    assert result["best_model_name"] in result["models"]
    assert set(result["metrics"]) == set(result["models"])
    for metrics in result["metrics"].values():
        assert 0 <= metrics["accuracy"] <= 1
        assert len(metrics["confusion_matrix"]) == 3


def test_saved_artifacts_produce_same_patient_prediction(tmp_path):
    frame = generate_dataset(120, save=False)
    result = train_and_compare(frame)
    model_name = result["best_model_name"]
    model = result["models"][model_name]
    preprocessor = result["preprocessor"]
    patient = frame.drop(columns=["Risk_Label", "Osteoporosis", "BMD_T_Score"]).iloc[0].to_dict()
    before = predict_patient(patient, model, preprocessor)

    model_path = tmp_path / "best_model.joblib"
    preprocessor_path = tmp_path / "preprocessor.joblib"
    save_artifacts(model, preprocessor, model_name, model_path, preprocessor_path)
    loaded_model, loaded_preprocessor, metadata = load_artifacts(model_path, preprocessor_path)
    after = predict_patient(patient, loaded_model, loaded_preprocessor)

    assert metadata["model_name"] == model_name
    assert before["risk_label"] == after["risk_label"]
    assert before["probabilities"] == after["probabilities"]