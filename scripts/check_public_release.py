#!/usr/bin/env python3
"""Fail on common secrets and machine-specific paths before public release."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "__pycache__", "output"}
TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cfg",
    ".csv",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
PATTERNS = {
    "macOS home-directory path": re.compile("/" + r"Users/[^/\s]+/"),
    "Linux home-directory path": re.compile("/" + r"home/[^/\s]+/"),
    "Windows home-directory path": re.compile(
        r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+[\\/]"
    ),
    "Dropbox-local path": re.compile(r"(?:/|\\)Dropbox(?:/|\\)"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "generic API secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*"
        r"['\"][^'\"]{8,}['\"]"
    ),
}
BINARY_PATTERNS = {
    label: re.compile(
        pattern.pattern.encode("utf-8"), pattern.flags & ~re.UNICODE
    )
    for label, pattern in PATTERNS.items()
    if label != "generic API secret"
}


def candidate_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in SKIP_PARTS for part in path.relative_to(ROOT).parts)
        and path.suffix.lower() in TEXT_SUFFIXES
    )


def scan_bytes(
    findings: list[str],
    label_path: str,
    payload: bytes,
) -> None:
    for label, pattern in BINARY_PATTERNS.items():
        if pattern.search(payload):
            findings.append(f"{label_path}: {label}")


def scan_binary_files(findings: list[str]) -> int:
    checked = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(
            part in SKIP_PARTS for part in path.relative_to(ROOT).parts
        ):
            continue
        relative = str(path.relative_to(ROOT))
        if path.suffix.lower() == ".npz":
            checked += 1
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    scan_bytes(
                        findings,
                        f"{relative}:{member.filename}",
                        archive.read(member),
                    )
        elif path.suffix.lower() in {".pdf", ".png"}:
            checked += 1
            scan_bytes(findings, relative, path.read_bytes())
    return checked


def main() -> None:
    findings: list[str] = []
    text_files = candidate_files()
    for path in text_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(ROOT)}:{line}: {label}")
    binary_count = scan_binary_files(findings)
    if findings:
        raise SystemExit(
            "Public-release scan found possible hazards:\n" + "\n".join(findings)
        )
    print(
        "Public-release scan passed "
        f"({len(text_files)} text files and {binary_count} binary files checked)."
    )


if __name__ == "__main__":
    main()
