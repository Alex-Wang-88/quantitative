from __future__ import annotations

from quantpaper.data import make_demo_provider
from quantpaper.features import FeatureEngine
from quantpaper.model import LightGBMModel


def test_lightgbm_artifact_has_time_ordered_validation_and_can_reload(tmp_path) -> None:
    provider = make_demo_provider(seed=7, quote_steps=2, replay_days=2)
    engine = FeatureEngine()
    training = engine.build_training_frame(provider.get_daily_frame(), horizon_days=5)

    model = LightGBMModel()
    artifact = model.fit(
        training,
        artifact_dir=tmp_path,
        min_training_rows=500,
        validation_fraction=0.2,
        min_validation_days=20,
    )

    assert artifact.training_rows >= 500
    assert artifact.validation_rows > 0
    assert artifact.validation_start is not None
    assert artifact.training_end < artifact.validation_start
    assert "validation_rank_corr" in artifact.metrics
    assert artifact.artifact_path is not None

    loaded = LightGBMModel.load(
        artifact.artifact_path,
        expected_feature_columns=engine.FACTOR_COLUMNS,
    )
    scored = loaded.score(engine.latest(provider.get_daily_frame()))
    assert len(scored) == len(provider.get_security_master())
    assert scored["model_version"].eq(artifact.model_version).all()
