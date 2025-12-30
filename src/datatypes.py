"""Configuration dataclasses for frame comparison tool."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union


class OddGeometryPolicy(str, Enum):
    """Policies controlling how odd-pixel geometry is handled for screenshots."""

    AUTO = "auto"
    FORCE_FULL_CHROMA = "force_full_chroma"
    SUBSAMP_SAFE = "subsamp_safe"


class RGBDither(str, Enum):
    """Dithering strategies used when converting RGB outputs to 8-bit."""

    ERROR_DIFFUSION = "error_diffusion"
    ORDERED = "ordered"
    NONE = "none"


class ExportRange(str, Enum):
    """Output range options for exported RGB PNGs."""

    FULL = "full"
    LIMITED = "limited"


class AnalysisThresholdMode(str, Enum):
    """Strategies for categorising frames by brightness."""

    QUANTILE = "quantile"
    FIXED_RANGE = "fixed_range"


class AutoLetterboxCropMode(str, Enum):
    """Auto letterbox cropping heuristics used during screenshot planning."""

    OFF = "off"
    BASIC = "basic"
    STRICT = "strict"


@dataclass
class AnalysisThresholds:
    """Configuration values for dark/bright frame detection."""

    mode: AnalysisThresholdMode = AnalysisThresholdMode.QUANTILE
    dark_quantile: float = 0.20
    bright_quantile: float = 0.80
    dark_luma_min: float = 0.062746
    dark_luma_max: float = 0.38
    bright_luma_min: float = 0.45
    bright_luma_max: float = 0.80


@dataclass
class AnalysisConfig:
    """Options controlling frame analysis, selection, and data caching."""

    frame_count_dark: int = 20
    frame_count_bright: int = 10
    frame_count_motion: int = 15
    user_frames: List[int] = field(default_factory=list)
    random_frames: int = 15
    save_frames_data: bool = True
    downscale_height: int = 480
    step: int = 2
    analyze_in_sdr: bool = True
    thresholds: AnalysisThresholds = field(default_factory=AnalysisThresholds)
    motion_use_absdiff: bool = False
    motion_scenecut_quantile: float = 0.0
    screen_separation_sec: int = 6
    motion_diff_radius: int = 4
    analyze_clip: str = ""
    random_seed: int = 20202020
    frame_data_filename: str = "generated.compframes"
    ignore_lead_seconds: float = 0.0
    ignore_trail_seconds: float = 0.0
    min_window_seconds: float = 5.0


@dataclass
class ScreenshotConfig:
    """Screenshot export behavior, renderer selection, and geometry tweaks."""

    directory_name: str = "screens"
    add_frame_info: bool = True
    use_ffmpeg: bool = False
    compression_level: int = 1
    upscale: bool = True
    single_res: int = 0
    mod_crop: int = 2
    letterbox_pillarbox_aware: bool = True
    auto_letterbox_crop: AutoLetterboxCropMode | str | bool = "off"
    pad_to_canvas: str = "off"
    letterbox_px_tolerance: int = 8
    center_pad: bool = True
    ffmpeg_timeout_seconds: float = 120.0
    odd_geometry_policy: OddGeometryPolicy = OddGeometryPolicy.AUTO
    rgb_dither: RGBDither = RGBDither.ERROR_DIFFUSION
    export_range: ExportRange = ExportRange.FULL


@dataclass
class ColorConfig:
    """HDR tonemapping, overlay, and verification controls."""

    enable_tonemap: bool = True
    preset: str = "reference"
    tone_curve: str = "bt.2390"
    dynamic_peak_detection: bool = True
    target_nits: float = 100.0
    dst_min_nits: float = 0.18
    overlay_enabled: bool = True
    overlay_text_template: str = (
        "Tonemapping Algorithm: {tone_curve} dpd = {dynamic_peak_detection} dst = {target_nits} nits"
    )
    overlay_mode: str = "minimal"
    verify_enabled: bool = True
    verify_frame: Optional[int] = None
    verify_auto: bool = True
    verify_start_seconds: float = 10.0
    verify_step_seconds: float = 10.0
    verify_max_seconds: float = 90.0
    verify_luma_threshold: float = 0.10
    strict: bool = False
    default_matrix_hd: Optional[str] = None
    default_matrix_sd: Optional[str] = None
    default_primaries_hd: Optional[str] = None
    default_primaries_sd: Optional[str] = None
    default_transfer_sdr: Optional[str] = None
    default_range_sdr: Optional[str] = None
    color_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    debug_color: bool = False
    knee_offset: float = 0.5
    dpd_preset: str = "high_quality"
    dpd_black_cutoff: float = 0.01
    post_gamma_enable: bool = False
    post_gamma: float = 1.0
    smoothing_period: float = 45.0
    scene_threshold_low: float = 0.8
    scene_threshold_high: float = 2.4
    percentile: float = 99.995
    contrast_recovery: float = 0.3
    metadata: str | int = "auto"
    use_dovi: Optional[bool] = None
    visualize_lut: bool = False
    show_clipping: bool = False
    _provided_keys: Set[str] = field(default_factory=set)


@dataclass
class SlowpicsConfig:
    """slow.pics upload automation flags and metadata."""

    auto_upload: bool = False
    collection_name: str = ""
    collection_suffix: str = ""
    is_hentai: bool = False
    is_public: bool = True
    tmdb_id: str = ""
    tmdb_category: str = ""
    remove_after_days: int = 0
    webhook_url: str = ""
    open_in_browser: bool = True
    create_url_shortcut: bool = True
    delete_screen_dir_after_upload: bool = True
    image_upload_timeout_seconds: float = 180.0


@dataclass
class TMDBConfig:
    """Configuration controlling TMDB lookup and caching behaviour."""

    api_key: str = ""
    unattended: bool = True
    confirm_matches: bool = False
    year_tolerance: int = 2
    enable_anime_parsing: bool = True
    cache_ttl_seconds: int = 86400
    cache_max_entries: int = 256
    category_preference: Optional[str] = None


@dataclass
class NamingConfig:
    """Filename parsing and display preferences."""

    always_full_filename: bool = True
    prefer_guessit: bool = True


@dataclass
class CLIProgressConfig:
    """Presentation preferences for progress indicators."""

    style: str = "fill"


@dataclass
class CLIConfig:
    """CLI presentation controls."""

    emit_json_tail: bool = True
    progress: CLIProgressConfig = field(default_factory=CLIProgressConfig)


@dataclass
class RunnerConfig:
    """Runner-specific toggles kept for legacy compatibility."""

    enable_service_mode: bool = True


@dataclass
class PathsConfig:
    """Filesystem paths configured by the user."""

    input_dir: str = "comparison_videos"


@dataclass
class RuntimeConfig:
    """Runtime safeguards such as memory limits."""

    ram_limit_mb: int = 8000
    vapoursynth_python_paths: List[str] = field(default_factory=list)
    force_reprobe: bool = False


@dataclass
class SourceConfig:
    """Preferred VapourSynth source plugin selection."""

    preferred: str = "lsmas"


@dataclass
class ReportConfig:
    """HTML report generation controls."""

    enable: bool = False
    open_after_generate: bool = True
    output_dir: str = "report"
    title: Optional[str] = None
    default_left_label: Optional[str] = None
    default_right_label: Optional[str] = None
    include_metadata: str = "minimal"
    thumb_height: int = 0
    default_mode: str = "slider"


@dataclass
class DiagnosticsConfig:
    """Diagnostic overlay computation controls."""

    per_frame_nits: bool = False


@dataclass
class AudioAlignmentConfig:
    """Audio-based offset estimation to help align clips before analysis."""

    enable: bool = False
    reference: str = ""
    use_vspreview: bool = False
    vspreview_mode: str = "baseline"
    show_suggested_in_preview: bool = True
    prompt_reuse_offsets: bool = False
    sample_rate: int = 16000
    hop_length: int = 512
    start_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    correlation_threshold: float = 0.55
    max_offset_seconds: float = 12.0
    offsets_filename: str = "generated.audio_offsets.toml"
    random_seed: int = 2025


@dataclass
class OverridesConfig:
    """Clip-specific overrides for trimming and frame rate adjustments."""

    trim: Dict[str, int] = field(default_factory=dict)
    trim_end: Dict[str, int] = field(default_factory=dict)
    change_fps: Dict[str, Union[List[int], str]] = field(default_factory=dict)


@dataclass
class AppConfig:
    """Aggregated configuration loaded from the user-provided TOML file."""

    analysis: AnalysisConfig
    screenshots: ScreenshotConfig
    cli: CLIConfig
    runner: RunnerConfig
    slowpics: SlowpicsConfig
    tmdb: TMDBConfig
    naming: NamingConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    overrides: OverridesConfig
    color: ColorConfig
    source: SourceConfig
    audio_alignment: AudioAlignmentConfig
    report: ReportConfig
    diagnostics: DiagnosticsConfig
