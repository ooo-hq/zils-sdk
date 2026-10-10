"""Customer training HTTP workflow. Execution stays in the hosted platform."""

import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import httpx

from . import account, training_data
from ._files import atomic_json, file_lock
from ._input import parse_json
from ._training_http import TrainingError, read_json
from .credentials import config_path

MODEL_ID = "jevk5-4b-v0.3"
ARTIFACTS = ("adapter_config.json", "adapter_model.safetensors", "model.json", "release.json")
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_RELEASE_BYTES = 1024 * 1024
TRANSFER_TIMEOUT = 120
UUID_PATTERN = r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}"
STATUSES = {
    "uploading",
    "validating",
    "awaiting_approval",
    "queued",
    "running",
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
        for field in ("model", "workflow", "result"):
            if job.get(field) is not None and not isinstance(job[field], dict):
                raise ValueError("Invalid job data")
        data = job.get("result") or {}
        for field in ("delivery", "baseline"):
            if data.get(field) is not None and not isinstance(data[field], dict):
                raise ValueError("Invalid evaluation")
        if "miners" in data and (
            not isinstance(data["miners"], list)
            or any(not isinstance(row, dict) for row in data["miners"])
        ):
            raise ValueError("Invalid evaluation")
        return job
    except (KeyError, ValueError, TypeError):
        raise ValueError("The training service returned an invalid job response.") from None


def public_status(job):
    job = _job({"job": job})
    result = {
        "id": job["id"],
        "name": job["name"],
        "status": job["status"],
        "download_available": False,
    }
    workflow = job.get("workflow") or {}
    if isinstance(workflow, dict) and workflow.get("state") in {
        "waiting_capacity",
        "waiting_worker",
        "needs_review",
        "activating",
        "activation_failed",
        "ready",
    }:
        result["workflow"] = workflow["state"]
    if job["status"] == "failed":
        result["cancelled"] = job.get("error") == "Cancelled by customer."
    if job["status"] == "completed":
        delivery = (job.get("result") or {}).get("delivery") or {}
        data = job.get("result") or {}
        candidate = next(
            (row for row in data.get("miners", []) if row.get("uid") == delivery.get("uid")), {}
        )
        evaluation = {}
        for label, row in (("baseline", data.get("baseline") or {}), ("candidate", candidate)):
            metrics = {}
            for key in ("accuracy", "brier", "skill"):
                if key in row:
                    value = row[key]
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError(
                            "The training service returned invalid evaluation metrics."
                        )
                    metrics[key] = value
            if metrics:
                evaluation[label] = metrics
        if evaluation:
            result["evaluation"] = evaluation
        if delivery.get("status") in ("accepted", "no_qualifying_model"):
            result["delivery"] = delivery["status"]
        if delivery.get("status") == "accepted":
            result["download_available"] = True
            if workflow.get("state") == "ready":
                model_id = workflow.get("model_id")
                if isinstance(model_id, str) and re.fullmatch(r"[a-zA-Z0-9._-]{1,256}", model_id):
                    result["model_id"] = model_id
    return result


def _download_file(client, grant, target, maximum, deadline):
    digest, size = hashlib.sha256(), 0
    try:
        with client.stream(
            "GET", grant["url"], headers={"Accept-Encoding": "identity"}
        ) as response:
            if (
                response.status_code != 200
                or response.headers.get("Content-Encoding", "identity") != "identity"
            ):
                raise ValueError("Unexpected storage response")
            length = response.headers.get("Content-Length")
            if length is not None and (
                not re.fullmatch(r"[0-9]+", length) or int(length) > maximum
            ):
                raise ValueError("Invalid download length")
            with os.fdopen(
                os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
            ) as output:
                for chunk in response.iter_raw(chunk_size=64 * 1024):
                    size += len(chunk)
                    if size > maximum or time.monotonic() > deadline:
                        raise ValueError("Download exceeded its limit")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if length is not None and size != int(length):
                raise ValueError("Truncated download")
            return digest.hexdigest(), size
    except (httpx.HTTPError, ValueError):
        raise TrainingError(
            None,
            "Model download could not be verified. No files were published and no request was retried.",
        ) from None


def verify_checkpoint(directory, job, records, round_id):
    try:
        delivery = job["result"]["delivery"]
        model = job["model"]
        digest = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()
        release = parse_json((directory / "release.json").read_text(encoding="utf-8"))
        metadata = parse_json((directory / "model.json").read_text(encoding="utf-8"))
        if (
            digest != delivery["sha256"]
            or release["sha256"] != digest
            or release["status"] != "accepted"
            or release["job_id"] != job["id"]
            or release["round_id"] != round_id
            or release["model"] != model
            or release["base"] != model["base"]
            or release["base_revision"] != model["base_revision"]
            or model["id"] != MODEL_ID
            or model["base"] != "alibiserikbay/JevK5"
            or model["base_revision"] != "c4f7fdb3aeab5582336406e78d3bef11bf98833d"
        ):
            raise ValueError("Release identity mismatch")
        temperature = metadata["temperature"]
        if (
            set(metadata) - {"version", "model", "kind", "temperature", "calibration"}
            or metadata["version"] != "zils-checkpoint/v1"
            or metadata["model"] != MODEL_ID
            or metadata["kind"] != "adapter"
            or type(temperature) not in (int, float)
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("Invalid adapter metadata")
        return digest
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError(
            "Downloaded model does not match the accepted job, release identity or checkpoint digest. No files were published."
        ) from None


class TrainingClient:
    def __init__(self, *, session=None, root=None, timeout=30):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be a finite positive number of seconds.")
        self.timeout = timeout
        self._session = session
        self.root = Path(root) if root is not None else config_path().resolve().parent / "training"

    def _connect(self):
        # Delay authentication until submit has validated and frozen local inputs.
        return self._session if self._session is not None else account.credentials()

    def _request(self, session, method, path, body=None):
        return read_json(
            method,
            session["base_url"] + path,
            token=session["access_token"],
            body=body,
            timeout=self.timeout,
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
            deadline = time.monotonic() + min(120, self.timeout)
            try:
                with (
                    Path(info["path"]).open("rb") as source,
                    httpx.Client(follow_redirects=False, timeout=min(120, self.timeout)) as client,
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

    def cancel(self, job_id):
        account.identifier(job_id)
        return public_status(
            _job(self._request(self._connect(), "POST", f"/v1/jobs/{job_id}/cancel", {}), job_id)
        )

    def download(self, job_id, destination):
        account.identifier(job_id)
        destination = Path(destination).expanduser().absolute()
        if destination.exists() or destination.is_symlink():
            raise ValueError(
                "Output already exists. Choose a new directory; existing files are never overwritten."
            )
        session = self._connect()
        job = _job(self._request(session, "GET", f"/v1/jobs/{job_id}"), job_id)
        if (
            job["status"] != "completed"
            or (job.get("result") or {}).get("delivery", {}).get("status") != "accepted"
            or (job.get("model") or {}).get("id") != MODEL_ID
        ):
            raise ValueError("This job has no accepted JevK5 model to download.")
        grants = self._request(session, "GET", f"/v1/jobs/{job_id}/downloads").get("downloads")
        if not isinstance(grants, dict) or set(grants) != set(ARTIFACTS):
            raise ValueError("The service returned an unexpected model file list.")
        rounds = set()
        for name, grant in grants.items():
            validate_grant(
                grant, session["config"], job_id=job_id, filename=name, use="model-download"
            )
            rounds.add(urlsplit(grant["url"]).path.split("/")[-2])
        if len(rounds) != 1:
            raise ValueError("Model files belong to different releases.")
        with tempfile.TemporaryDirectory(
            prefix=".zils-download-", dir=destination.parent
        ) as temporary:
            directory = Path(temporary)
            records, size = [], 0
            deadline = time.monotonic() + 4 * TRANSFER_TIMEOUT
            with httpx.Client(
                follow_redirects=False, timeout=min(TRANSFER_TIMEOUT, self.timeout)
            ) as client:
                for name in ARTIFACTS:
                    maximum = (
                        MAX_RELEASE_BYTES
                        if name == "release.json"
                        else min(
                            16384 if name == "model.json" else MAX_ARTIFACT_BYTES,
                            MAX_ARTIFACT_BYTES - size,
                        )
                    )
                    digest, count = _download_file(
                        client,
                        grants[name],
                        directory / name,
                        maximum,
                        min(deadline, time.monotonic() + min(TRANSFER_TIMEOUT, self.timeout)),
                    )
                    if name != "release.json":
                        size += count
                        records.append([name, digest])
            digest = verify_checkpoint(directory, job, records, next(iter(rounds)))
            # mkdir and hard links fail if another process created an output; nothing is replaced.
            destination.mkdir(mode=0o700)
            published = []
            try:
                for name in ARTIFACTS:
                    os.link(directory / name, destination / name)
                    published.append(destination / name)
            except BaseException:
                for file in published:
                    file.unlink(missing_ok=True)
                if not any(destination.iterdir()):
                    destination.rmdir()
                raise
        return {
            "id": job_id,
            "directory": str(destination),
            "files": list(ARTIFACTS),
            "sha256": digest,
        }
