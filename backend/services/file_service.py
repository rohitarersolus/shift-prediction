from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from backend.config import settings


class AppError(Exception):
    def __init__(self, message: str, status_code: int = 400, extra: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.extra = extra or {}

    @property
    def response_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"detail": self.message}
        body.update(self.extra)
        return body


@dataclass
class SavedUpload:
    original_name: str
    extension: str
    path: Path
    size_bytes: int


def get_file_extension(filename: str | None) -> str:
    if not filename:
        raise AppError("Please choose a file before running prediction.")

    extension = Path(filename).suffix.lower()
    if extension not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise AppError(f"Unsupported file type. Please upload one of: {allowed}.")

    return extension


async def save_upload_file(upload: UploadFile) -> SavedUpload:
    extension = get_file_extension(upload.filename)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target_path = settings.uploads_dir / f"{timestamp}-{uuid4().hex}{extension}"

    size_bytes = 0
    try:
        with target_path.open("wb") as output_file:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break

                size_bytes += len(chunk)
                if size_bytes > settings.max_upload_size_bytes:
                    raise AppError(
                        f"File is too large. Maximum supported size is {settings.max_upload_size_mb} MB."
                    )

                output_file.write(chunk)
    except Exception:
        delete_file(target_path)
        raise
    finally:
        await upload.close()

    return SavedUpload(
        original_name=upload.filename or target_path.name,
        extension=extension,
        path=target_path,
        size_bytes=size_bytes,
    )


def delete_file(path: Path | None) -> None:
    if path and path.exists():
        path.unlink(missing_ok=True)
