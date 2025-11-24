from types import SimpleNamespace

from src.frame_compare.vs.tonemap import _resolve_tonemap_settings


def test_resolve_tonemap_auto_enable_with_rpu():
    """Test that use_dovi becomes True when set to auto (None) and RPU blob is present."""
    cfg = SimpleNamespace(
        preset="custom",
        tone_curve="bt.2390",
        target_nits=100.0,
        dynamic_peak_detection=True,
        dst_min_nits=0.18,
        knee_offset=0.5,
        dpd_preset="high_quality",
        dpd_black_cutoff=0.01,
        smoothing_period=45.0,
        scene_threshold_low=0.8,
        scene_threshold_high=2.4,
        percentile=99.995,
        contrast_recovery=0.3,
        metadata="auto",
        use_dovi="auto",  # Explicitly auto
        visualize_lut=False,
        show_clipping=False,
        _provided_keys=set(),
    )

    # Case 1: No props -> use_dovi remains None (auto)
    settings = _resolve_tonemap_settings(cfg, props={})
    assert settings.use_dovi is None

    # Case 2: Props with RPU blob -> use_dovi becomes True
    props = {"DolbyVisionRPU": b"some_data"}
    settings = _resolve_tonemap_settings(cfg, props=props)
    assert settings.use_dovi is True

    # Case 3: Props with alternative RPU key -> use_dovi becomes True
    props = {"_DolbyVisionRPU": b"some_data"}
    settings = _resolve_tonemap_settings(cfg, props=props)
    assert settings.use_dovi is True

    # Case 4: Props without RPU -> use_dovi remains None
    props = {"_Matrix": 1}
    settings = _resolve_tonemap_settings(cfg, props=props)
    assert settings.use_dovi is None


def test_resolve_tonemap_explicit_override():
    """Test that explicit use_dovi settings are respected regardless of RPU presence."""
    cfg = SimpleNamespace(
        preset="custom",
        use_dovi="false",  # Explicitly false
        _provided_keys=set(),
    )

    # Even with RPU, should be False
    props = {"DolbyVisionRPU": b"some_data"}
    settings = _resolve_tonemap_settings(cfg, props=props)
    assert settings.use_dovi is False

    cfg.use_dovi = "true"
    # With RPU, should be True
    settings = _resolve_tonemap_settings(cfg, props=props)
    assert settings.use_dovi is True

    # Without RPU, should still be True (forced on)
    settings = _resolve_tonemap_settings(cfg, props={})
    assert settings.use_dovi is True
