from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    allowed_extensions: tuple[str, ...] = (".csv", ".xls", ".xlsx")
    max_upload_size_mb: int = 25
    cors_origins: tuple[str, ...] = (
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    )

    @property
    def backend_dir(self) -> Path:
        return self.base_dir / "backend"

    @property
    def frontend_dir(self) -> Path:
        return self.base_dir / "frontend"

    @property
    def frontend_assets_dir(self) -> Path:
        return self.frontend_dir / "assets"

    @property
    def uploads_dir(self) -> Path:
        return self.base_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "outputs"

    @property
    def engine_package_dir(self) -> Path:
        return self.base_dir / "shift_engine_package_v3_day_first_reader"

    @property
    def engine_artifacts_dir(self) -> Path:
        return self.engine_package_dir / "artifacts"

    @property
    def engine_version(self) -> str:
        return "v3_day_first_reader"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()


def ensure_runtime_directories() -> None:
    for directory in (
        settings.uploads_dir,
        settings.outputs_dir,
        settings.frontend_dir,
        settings.frontend_assets_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
