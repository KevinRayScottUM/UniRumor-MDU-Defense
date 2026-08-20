"""Streaming validation for the narrow Task07 multipart submission contract."""

from __future__ import annotations

import ntpath
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Set

from fastapi import Request
from python_multipart import MultipartParser
from python_multipart.exceptions import FormParserError
from python_multipart.multipart import parse_options_header

from .workspace import ALLOWED_INPUT_EXTENSIONS, WebWorkspaceManager


MAX_CLAIM_CODEPOINTS = 2000
MAX_CLAIM_UTF8_BYTES = MAX_CLAIM_CODEPOINTS * 4
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
VIDEO_MIME_BY_EXTENSION = {
    ".mp4": frozenset(("video/mp4",)),
    ".m4v": frozenset(("video/x-m4v", "video/mp4")),
    ".mov": frozenset(("video/quicktime",)),
    ".webm": frozenset(("video/webm",)),
}


class SubmissionValidationError(ValueError):
    """Stable internal mapping for a rejected public submission."""

    def __init__(self, status_code: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class SubmissionHeaders:
    boundary: bytes
    maximum_request_bytes: int


@dataclass(frozen=True)
class ParsedSubmission:
    claim: str
    video_path: Path


def _reject(status_code: int, code: str) -> None:
    raise SubmissionValidationError(status_code, code)


def validate_submission_headers(
    request: Request,
    max_upload_bytes: int,
) -> SubmissionHeaders:
    """Validate cheap HTTP metadata before reserving bounded capacity."""

    content_type = request.headers.get("content-type")
    if content_type is None:
        _reject(400, "malformed_request")
    try:
        media_type, parameters = parse_options_header(content_type)
    except (TypeError, ValueError):
        _reject(400, "malformed_request")
    if media_type.lower() != b"multipart/form-data":
        _reject(400, "malformed_request")
    boundary = parameters.get(b"boundary")
    if not boundary:
        _reject(400, "malformed_request")

    maximum_request_bytes = max_upload_bytes + MAX_MULTIPART_OVERHEAD_BYTES
    raw_lengths = [
        value
        for name, value in request.scope.get("headers", ())
        if name.lower() == b"content-length"
    ]
    if len(raw_lengths) > 1:
        _reject(400, "malformed_request")
    if raw_lengths:
        try:
            raw_length = raw_lengths[0].decode("ascii")
            if not raw_length.isdigit():
                raise ValueError
            content_length = int(raw_length)
        except (UnicodeDecodeError, ValueError):
            _reject(400, "malformed_request")
        if content_length > maximum_request_bytes:
            _reject(413, "upload_too_large")

    return SubmissionHeaders(
        boundary=boundary,
        maximum_request_bytes=maximum_request_bytes,
    )


def _validated_filename(raw_filename: bytes) -> tuple[str, str]:
    try:
        filename = raw_filename.decode("utf-8")
    except UnicodeDecodeError:
        _reject(422, "invalid_filename")
    if (
        not filename
        or "\x00" in filename
        or "/" in filename
        or "\\" in filename
        or filename in {".", ".."}
        or Path(filename).is_absolute()
        or bool(ntpath.splitdrive(filename)[0])
    ):
        _reject(422, "invalid_filename")
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_INPUT_EXTENSIONS:
        _reject(415, "unsupported_video_type")
    return filename, extension


def _valid_iso_bmff_ftyp(probe: bytes, total_size: int) -> bool:
    """Validate the complete declared structure of the leading ``ftyp`` box."""

    if len(probe) < 16 or total_size < 16 or probe[4:8] != b"ftyp":
        return False

    size32 = int.from_bytes(probe[:4], "big")
    header_size = 8
    if size32 == 1:
        header_size = 16
        if len(probe) < 24 or total_size < 24:
            return False
        box_size = int.from_bytes(probe[8:16], "big")
    elif size32 == 0:
        box_size = total_size
    else:
        box_size = size32

    minimum_size = header_size + 8
    if box_size < minimum_size or box_size > total_size:
        return False
    return (box_size - minimum_size) % 4 == 0


def _valid_container_signature(
    extension: str,
    probe: bytes,
    total_size: int,
) -> bool:
    if extension == ".webm":
        return probe.startswith(b"\x1a\x45\xdf\xa3")
    if extension in {".mp4", ".m4v", ".mov"}:
        return _valid_iso_bmff_ftyp(probe, total_size)
    return False


class _StreamingSubmissionReceiver:
    def __init__(
        self,
        workspace: WebWorkspaceManager,
        job_id: str,
        max_upload_bytes: int,
    ) -> None:
        self._workspace = workspace
        self._job_id = job_id
        self._max_upload_bytes = max_upload_bytes
        self._headers: Dict[bytes, bytes] = {}
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._part_kind: Optional[str] = None
        self._seen_parts: Set[str] = set()
        self._claim_bytes = bytearray()
        self._video_file: Optional[BinaryIO] = None
        self._video_path: Optional[Path] = None
        self._video_extension: Optional[str] = None
        self._video_size = 0
        self._video_probe = bytearray()
        self.ended = False

    @property
    def callbacks(self) -> dict:
        return {
            "on_part_begin": self.on_part_begin,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
            "on_header_begin": self.on_header_begin,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_end": self.on_end,
        }

    def on_part_begin(self) -> None:
        self._headers = {}
        self._part_kind = None

    def on_header_begin(self) -> None:
        self._header_field = bytearray()
        self._header_value = bytearray()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field.extend(data[start:end])

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value.extend(data[start:end])

    def on_header_end(self) -> None:
        field = bytes(self._header_field).strip().lower()
        value = bytes(self._header_value).strip()
        if not field or field in self._headers:
            _reject(400, "malformed_request")
        self._headers[field] = value

    def on_headers_finished(self) -> None:
        disposition = self._headers.get(b"content-disposition")
        if disposition is None:
            _reject(400, "malformed_request")
        try:
            disposition_type, parameters = parse_options_header(disposition)
        except (TypeError, ValueError):
            _reject(400, "malformed_request")
        if disposition_type.lower() != b"form-data":
            _reject(400, "malformed_request")
        raw_name = parameters.get(b"name")
        try:
            name = None if raw_name is None else raw_name.decode("ascii")
        except UnicodeDecodeError:
            _reject(400, "malformed_request")
        if name not in {"claim", "video"} or name in self._seen_parts:
            _reject(400, "malformed_request")
        self._seen_parts.add(name)
        self._part_kind = name

        if name == "claim":
            if b"filename" in parameters:
                _reject(400, "malformed_request")
            return

        raw_filename = parameters.get(b"filename")
        if raw_filename is None:
            _reject(422, "invalid_filename")
        _, extension = _validated_filename(raw_filename)
        raw_mime = self._headers.get(b"content-type")
        if raw_mime is None:
            _reject(415, "unsupported_video_type")
        try:
            mime = raw_mime.decode("ascii").lower()
        except UnicodeDecodeError:
            _reject(415, "unsupported_video_type")
        if mime not in VIDEO_MIME_BY_EXTENSION[extension]:
            _reject(415, "unsupported_video_type")

        self._video_extension = extension
        self._video_path = self._workspace.job_input_path(
            self._job_id,
            extension,
        )
        self._video_file = self._workspace.create_job_input(
            self._job_id,
            extension,
        )

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self._part_kind == "claim":
            if len(self._claim_bytes) + len(chunk) > MAX_CLAIM_UTF8_BYTES:
                _reject(422, "invalid_claim")
            self._claim_bytes.extend(chunk)
            return
        if self._part_kind != "video" or self._video_file is None:
            _reject(400, "malformed_request")
        if self._video_size + len(chunk) > self._max_upload_bytes:
            _reject(413, "upload_too_large")
        self._video_size += len(chunk)
        remaining_probe = 64 - len(self._video_probe)
        if remaining_probe > 0:
            self._video_probe.extend(chunk[:remaining_probe])
        self._video_file.write(chunk)

    def on_part_end(self) -> None:
        if self._part_kind == "video" and self._video_file is not None:
            self._video_file.close()
            self._video_file = None
        self._part_kind = None

    def on_end(self) -> None:
        self.ended = True

    def close(self) -> None:
        if self._video_file is not None:
            self._video_file.close()
            self._video_file = None

    def finish(self) -> ParsedSubmission:
        if "video" in self._seen_parts and "claim" not in self._seen_parts:
            _reject(422, "invalid_claim")
        if self._seen_parts != {"claim", "video"} or self._video_path is None:
            _reject(400, "malformed_request")
        try:
            claim = self._claim_bytes.decode("utf-8")
        except UnicodeDecodeError:
            _reject(422, "invalid_claim")
        if not claim.strip() or len(claim) > MAX_CLAIM_CODEPOINTS:
            _reject(422, "invalid_claim")
        if self._video_size == 0:
            _reject(422, "empty_upload")
        if self._video_extension is None or not _valid_container_signature(
            self._video_extension,
            bytes(self._video_probe),
            self._video_size,
        ):
            _reject(415, "unsupported_video_type")
        return ParsedSubmission(claim=claim, video_path=self._video_path)


async def receive_submission(
    request: Request,
    headers: SubmissionHeaders,
    workspace: WebWorkspaceManager,
    job_id: str,
    max_upload_bytes: int,
) -> ParsedSubmission:
    """Incrementally parse and store one strict claim/video request."""

    receiver = _StreamingSubmissionReceiver(
        workspace,
        job_id,
        max_upload_bytes,
    )
    try:
        parser = MultipartParser(
            headers.boundary,
            receiver.callbacks,
            max_size=headers.maximum_request_bytes,
        )
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > headers.maximum_request_bytes:
                _reject(413, "upload_too_large")
            if parser.write(chunk) != len(chunk):
                _reject(413, "upload_too_large")
        parser.finalize()
    except SubmissionValidationError:
        raise
    except FormParserError:
        _reject(400, "malformed_request")
    finally:
        receiver.close()

    if not receiver.ended:
        _reject(400, "malformed_request")
    return receiver.finish()


__all__ = [
    "MAX_CLAIM_CODEPOINTS",
    "ParsedSubmission",
    "SubmissionHeaders",
    "SubmissionValidationError",
    "VIDEO_MIME_BY_EXTENSION",
    "receive_submission",
    "validate_submission_headers",
]
