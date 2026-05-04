from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import unescape
from http.cookiejar import CookieJar
from pathlib import Path
import re
import shutil
import time
import urllib.parse
import urllib.request
import zipfile

import gdown

from v2.training.hf_utils import resolve_repo_path


FOLDERS: tuple[tuple[str, str], ...] = (
    ("train-left-image", "https://drive.google.com/drive/folders/1KN8BSF5KovPuNpKf0W2hScVpo70bRewI?usp=sharing"),
    ("train-right-image", "https://drive.google.com/drive/folders/1UG1U6iZVKsSk3Amn84bE1iFN53OKlsps?usp=sharing"),
    ("train-disparity", "https://drive.google.com/drive/folders/18obNjqFMzPuga6ZLN4UwCAqUjP7tQlKg?usp=sharing"),
)


@dataclass(frozen=True)
class DownloadItem:
    file_id: str
    name: str
    local_path: Path


class DriveDownloadError(RuntimeError):
    pass


class DriveQuotaExceededError(DriveDownloadError):
    pass


def _build_opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    return opener


def _parse_drive_warning_page(html: str) -> str | None:
    form_match = re.search(r'<form[^>]+action="([^"]+)"', html)
    if form_match is None:
        return None
    action = unescape(form_match.group(1))
    pairs = re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', html)
    params = {name: unescape(value) for name, value in pairs}
    if not params:
        return None
    return action + "?" + urllib.parse.urlencode(params)


def _stream_to_file(response, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    with temp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    temp_path.replace(destination)


def _classify_html_error(html: str, file_id: str) -> DriveDownloadError:
    preview = html[:300].replace("\n", " ")
    lowered = html.lower()
    if "quota exceeded" in lowered or "too many users have viewed or downloaded" in lowered:
        return DriveQuotaExceededError(f"Quota exceeded for {file_id}: {preview}")
    return DriveDownloadError(f"Unexpected HTML response for {file_id}: {preview}")


def download_google_drive_file(
    file_id: str,
    destination: Path,
    *,
    retries: int = 4,
    retry_delay_seconds: float = 20.0,
) -> None:
    if destination.exists() and zipfile.is_zipfile(destination):
        return

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        opener = _build_opener()
        initial_url = f"https://drive.google.com/uc?id={file_id}"
        try:
            with opener.open(initial_url) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    _stream_to_file(response, destination)
                    return
                html = response.read().decode("utf-8", "ignore")

            confirm_url = _parse_drive_warning_page(html)
            if confirm_url is None:
                raise DriveDownloadError(f"Could not parse Google Drive warning page for {file_id}")

            with opener.open(confirm_url) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    html = response.read(2000).decode("utf-8", "ignore")
                    raise _classify_html_error(html, file_id)
                _stream_to_file(response, destination)
                return
        except DriveQuotaExceededError as error:
            last_error = error
            if attempt == retries:
                break
            sleep_seconds = retry_delay_seconds * attempt
            print(f"    quota retry {attempt}/{retries} for {destination.name}; sleeping {sleep_seconds:.0f}s", flush=True)
            time.sleep(sleep_seconds)
        except Exception as error:
            last_error = error
            if attempt == retries:
                break
            sleep_seconds = min(10.0 * attempt, 30.0)
            print(f"    retry {attempt}/{retries} for {destination.name}; sleeping {sleep_seconds:.0f}s", flush=True)
            time.sleep(sleep_seconds)

    if destination.with_suffix(destination.suffix + ".part").exists():
        destination.with_suffix(destination.suffix + ".part").unlink()
    if last_error is None:
        raise DriveDownloadError(f"Download failed for {file_id}")
    raise last_error


def list_folder_items(folder_url: str, output_dir: Path) -> list[DownloadItem]:
    raw_items = gdown.download_folder(
        url=folder_url,
        output=str(output_dir),
        quiet=True,
        use_cookies=False,
        skip_download=True,
    )
    return [
        DownloadItem(file_id=item.id, name=item.path, local_path=resolve_repo_path(item.local_path))
        for item in raw_items
    ]


def extract_all_zips(stage_dir: Path, target_dir: Path) -> None:
    zips = sorted(stage_dir.glob("*.zip"))
    if not zips:
        return
    print(f"Extracting {len(zips)} zip files into {target_dir}", flush=True)
    for index, zip_path in enumerate(zips, start=1):
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(f"Invalid zip file: {zip_path}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target_dir)
        zip_path.unlink()
        if index % 5 == 0 or index == len(zips):
            print(f"  extracted {index}/{len(zips)}: {zip_path.name}", flush=True)


def write_failure_log(base_dir: Path, folder_name: str, failures: list[tuple[str, str]]) -> None:
    log_path = base_dir / f"{folder_name}_failed_downloads.txt"
    if not failures:
        if log_path.exists():
            log_path.unlink()
        return
    lines = [f"{name}\t{reason}" for name, reason in failures]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_folder(base_dir: Path, folder_name: str, folder_url: str) -> None:
    stage_dir = base_dir / "_downloads" / folder_name
    final_dir = base_dir / folder_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    items = list_folder_items(folder_url, stage_dir)
    print(f"Processing {folder_name}: {len(items)} sequence archives", flush=True)
    failures: list[tuple[str, str]] = []

    for index, item in enumerate(items, start=1):
        if item.local_path.exists() and zipfile.is_zipfile(item.local_path):
            print(f"  skip {index}/{len(items)}: {item.name}", flush=True)
            continue
        print(f"  download {index}/{len(items)}: {item.name}", flush=True)
        try:
            download_google_drive_file(item.file_id, item.local_path)
        except DriveDownloadError as error:
            message = str(error)
            failures.append((item.name, message))
            print(f"    failed: {message}", flush=True)

    extract_all_zips(stage_dir, final_dir)
    write_failure_log(base_dir, folder_name, failures)
    print(f"Completed {folder_name}: {sum(1 for _ in final_dir.glob('*'))} extracted files", flush=True)
    if failures:
        print(f"  pending failures in {folder_name}: {len(failures)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the DrivingStereo training data")
    parser.add_argument("--base-dir", default="v2/data/raw/driving_stereo")
    parser.add_argument(
        "--folders",
        nargs="*",
        default=[name for name, _ in FOLDERS],
        help="Subset of folders to process (default: all training folders)",
    )
    args = parser.parse_args()

    base_dir = resolve_repo_path(args.base_dir)
    selected = set(args.folders)
    for folder_name, folder_url in FOLDERS:
        if folder_name in selected:
            process_folder(base_dir, folder_name, folder_url)

    print("DrivingStereo download/extract complete", flush=True)


if __name__ == "__main__":
    main()