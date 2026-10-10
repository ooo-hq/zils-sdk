"""Customer training HTTP workflow. Execution stays in the hosted platform."""

import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import httpx

from . import account, training_data
from ._files import atomic_json, file_lock
from ._training_http import TrainingError, read_json
from .credentials import config_path

MODEL_ID = "jevk5-4b-v0.3"
ARTIFACTS = ("adapter_config.json", "adapter_model.safetensors", "model.json", "release.json")
UUID_PATTERN = r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}"
STATUSES = {
    "uploading",
    "validating",
    "awaiting_approval",
    "queued",
    "training",
    "evaluating",
    "completed",
    "failed",
}


def validate_grant(descriptor: dict, config: dict, *, job_id: str, filename: str, use: str) -> dict:
    account.identifier(job_id)
    try:
        upload = use == "dataset-upload"
        if use not in ("dataset-upload", "model-download") or filename not in (
            [s + ".jsonl" for s in training_data.SPLITS] if upload else ARTIFACTS
        ):
            raise ValueError("Invalid file")
        raw, provider = descriptor["url"], descriptor["provider"]
        if not isinstance(raw, str) or len(raw) > 16384 or re.search(r"[\\\x00-\x20\x7f#]", raw):
            raise ValueError("Invalid URL")
        parsed = urlsplit(raw)
        actual_origin = parsed.scheme + "://" + parsed.netloc
        if provider not in ("supabase", "spaces") or actual_origin != config["storage"].get(
            provider + "_origin"
        ):
            raise ValueError("Invalid provider")
        account.origin(actual_origin)
        if "%" in parsed.path or any(part in (".", "..") for part in parsed.path.split("/")):
            raise ValueError("Invalid raw path")
        prefix = (
            (r"/storage/v1/object/" + ("upload/sign/" if upload else "sign/"))
            if provider == "supabase"
            else rf"/objects/{UUID_PATTERN}/"
        )
        suffix = (
            f"fez-training-data/{job_id}/inputs/"
            if upload
            else rf"fez-training-models/{job_id}/releases/{UUID_PATTERN}/"
        )
        if not re.fullmatch(prefix + suffix + re.escape(filename), parsed.path):
            raise ValueError("Invalid path")
        pairs = parse_qsl(
            parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=30
        )
        query = dict(pairs)
        if len(pairs) != len(query) or not query:
            raise ValueError("Invalid grant")
        if provider == "spaces" and upload:
            if query.get("partNumber") != "1" or not query.get("uploadId"):
                raise ValueError("Invalid multipart upload")
        elif "partNumber" in query or "uploadId" in query:
            raise ValueError("Unexpected multipart grant")
        if provider == "supabase" and (set(query) != {"token"} or not query["token"]):
            raise ValueError("Invalid storage token")
        expiry = datetime.fromisoformat(descriptor["expires_at"].replace("Z", "+00:00"))
        if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc):
            raise ValueError("Expired grant")
        if upload:
            headers = {"Content-Type": "application/octet-stream"}
            if provider == "supabase":
                headers["x-upsert"] = "false"
            if (
                descriptor["method"] != "PUT"
                or descriptor["headers"] != headers
                or set(descriptor) != {"url", "provider", "expires_at", "method", "headers"}
            ):
                raise ValueError("Invalid upload headers")
        elif set(descriptor) != {"url", "provider", "expires_at"}:
            raise ValueError("Invalid download descriptor")
        return descriptor
    except (KeyError, ValueError, TypeError, AttributeError):
        raise ValueError(
            "Storage grant is expired or does not match the expected provider, job and file."
        ) from None


def _job(response, job_id=None):
    try:
        job = response["job"]
        account.identifier(job["id"])
        if job_id is not None and job["id"] != job_id or job["status"] not in STATUSES:
            raise ValueError("Unexpected job")
        if not isinstance(job.get("name"), str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9-]{0,63}", job["name"]
        ):
            raise ValueError("Invalid name")
        return job
    except (KeyError, ValueError, TypeError):
        raise ValueError("The training service returned an invalid job response.") from None


def public_status(job):
    job = _job({"job": job})
    result = {"id": job["id"], "name": job["name"], "status": job["status"]}
    return result


class TrainingClient:
    def __init__(self, *, session=None, root=None):
        self._session = session
        self.root = Path(root) if root is not None else config_path().resolve().parent / "training"

    def _connect(self):
        # Delay authentication until submit has validated and frozen local inputs.
        return self._session if self._session is not None else account.credentials()

    def _request(self, session, method, path, body=None):
        return read_json(
            method, session["base_url"] + path, token=session["access_token"], body=body
        )

    def _save(self, receipt):
        atomic_json(self.root / (receipt["job_id"] + ".json"), receipt)

    def _cleanup(self, snapshot):
        directory = self.root / "snapshots" / account.identifier(snapshot)
        if directory.is_dir() and not directory.is_symlink():
            shutil.rmtree(directory)

    def submit(self, name, files, policy, consent):
        if not isinstance(policy, dict) or set(policy) != {"min_accuracy", "min_brier_improvement"}:
            raise ValueError("Supply both training acceptance thresholds.")
        policy = training_data.validate_policy(
            name, policy["min_accuracy"], policy["min_brier_improvement"], consent
        )
        prepared = training_data.prepare(files, self.root / "snapshots")
        pending = self.root / ("pending-" + prepared["snapshot"] + ".json")
        retain = False
        try:
            session = self._connect()
            model = session["config"].get("model")
            if not isinstance(model, dict) or model.get("id") != MODEL_ID:
                raise ValueError("CLI training currently requires the JevK5 text-training profile.")
            receipt = {
                "version": 1,
                "job_id": None,
                "account_id": session["account_id"],
                "base_url": session["base_url"],
                "stage": "creating",
                **prepared,
            }
            atomic_json(pending, receipt)
            retain = True
            try:
                response = self._request(
                    session,
                    "POST",
                    "/v1/jobs",
                    {
                        "name": name,
                        "model": model,
                        "acceptance": policy,
                        "allow_training_data_export": True,
                    },
                )
                job = _job(response)
            except (TrainingError, ValueError) as error:
                if isinstance(error, TrainingError) and not error.uncertain:
                    pending.unlink(missing_ok=True)
                    retain = False
                raise TrainingError(
                    getattr(error, "status", None),
                    "Job creation could not be verified. Run zils train list before submitting again; no create request was retried.",
                    True,
                ) from None
            receipt.update(job_id=job["id"], stage="uploading")
            self._save(receipt)
            pending.unlink()
            print(
                f"Job created: {job['id']}. Resume this job if an upload is interrupted.",
                file=sys.stderr,
            )
            with file_lock(self.root / (job["id"] + ".lock")):
                return self._upload_and_submit(session, receipt, response)
        finally:
            if not retain:
                self._cleanup(prepared["snapshot"])

    def _upload_and_submit(self, session, receipt, response):
        job_id = receipt["job_id"]
        job = _job(response, job_id)
        if job["status"] != "uploading":
            receipt["stage"] = "submitted"
            self._save(receipt)
            self._cleanup(receipt["snapshot"])
            return public_status(job)
        uploads = response.get("uploads")
        if not isinstance(uploads, dict) or set(uploads) != set(training_data.SPLITS):
            raise ValueError("The training service returned invalid upload slots.")
        for split, grant in uploads.items():
            if (
                not isinstance(grant, dict)
                or set(grant) != {"uploaded"}
                or grant["uploaded"] is not True
            ):
                validate_grant(
                    grant,
                    session["config"],
                    job_id=job_id,
                    filename=split + ".jsonl",
                    use="dataset-upload",
                )
        for split in training_data.SPLITS:
            grant = uploads[split]
            if grant == {"uploaded": True}:
                continue
            info = receipt["files"][split]
            deadline = time.monotonic() + 120
            try:
                with (
                    Path(info["path"]).open("rb") as source,
                    httpx.Client(follow_redirects=False, timeout=120) as client,
                ):
                    with client.stream(
                        "PUT",
                        grant["url"],
                        headers={**grant["headers"], "Content-Length": str(info["size"])},
                        content=source,
                    ) as response:
                        if not 200 <= response.status_code < 300:
                            raise TrainingError(
                                response.status_code,
                                f"Dataset upload did not complete. Resume job {job_id}; no upload was retried.",
                                True,
                            )
                        size = 0
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > 64 * 1024 or time.monotonic() > deadline:
                                raise ValueError("Upload acknowledgement limit")
            except (httpx.HTTPError, ValueError):
                raise TrainingError(
                    None,
                    f"Dataset upload could not be verified. Resume job {job_id}; no upload was retried.",
                    True,
                ) from None
        receipt["stage"] = "submitting"
        self._save(receipt)
        result = _job(self._request(session, "POST", f"/v1/jobs/{job_id}/submit", {}), job_id)
        if result["status"] == "uploading":
            raise ValueError(f"Submission has not been confirmed. Resume job {job_id}.")
        receipt["stage"] = "submitted"
        self._save(receipt)
        self._cleanup(receipt["snapshot"])
        return public_status(result)

    def resume(self, job_id, files):
        account.identifier(job_id)
        prepared = training_data.prepare(files, self.root / "snapshots")
        retain = False
        try:
            session = self._connect()
            with file_lock(self.root / (job_id + ".lock")):
                receipt = training_data.load_receipt(
                    self.root, job_id, session["account_id"], session["base_url"]
                )
                if any(
                    prepared["files"][s]["sha256"] != receipt["files"][s]["sha256"]
                    for s in training_data.SPLITS
                ):
                    raise ValueError(
                        "Dataset bytes changed since this job was created. Resume requires the original files."
                    )
                job = _job(self._request(session, "GET", f"/v1/jobs/{job_id}"), job_id)
                if job["status"] != "uploading":
                    receipt["stage"] = "submitted"
                    self._save(receipt)
                    self._cleanup(receipt["snapshot"])
                    return public_status(job)
                previous_snapshot = receipt["snapshot"]
                receipt.update(prepared)
                self._save(receipt)
                retain = True
                self._cleanup(previous_snapshot)
                return self._upload_and_submit(
                    session,
                    receipt,
                    self._request(session, "POST", f"/v1/jobs/{job_id}/uploads", {}),
                )
        finally:
            if not retain:
                self._cleanup(prepared["snapshot"])

    def list(self):
        response = self._request(self._connect(), "GET", "/v1/jobs")
        try:
            return {
                key: [public_status(job) for job in response[key]] for key in ("jobs", "models")
            }
        except (KeyError, TypeError):
            raise ValueError("The training service returned an invalid job list.") from None

    def get(self, job_id):
        account.identifier(job_id)
        return public_status(
            _job(self._request(self._connect(), "GET", f"/v1/jobs/{job_id}"), job_id)
        )
