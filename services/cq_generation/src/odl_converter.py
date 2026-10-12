"""Adapter for opendataloader-pdf (JAR): PDF bytes in, conversion result bytes out.

The temp directory, JVM flags and collecting the result file all stay in here: the
router only gets bytes, and failures are told apart only by the two exceptions below.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from config import settings


class ConversionTimeout(Exception):
    """The conversion exceeded the time limit and java was killed."""


class ConversionFailed(Exception):
    """The conversion failed or produced no result file.

    The message holds only what is safe to put in a response body: java stderr leaks
    temp paths, so it is only logged here.
    """


def run_jar(input_path: Path, output_dir: Path) -> None:
    """Calls the opendataloader-pdf JAR directly.

    The package's convert() is not used because it offers no way to set a timeout: its
    internal runner calls subprocess.run without timeout=. subprocess.run kills the
    child java itself on timeout (wrapping it in a thread/process pool leaves the JVM as
    a grandchild that survives).
    """
    from importlib.resources import as_file, files

    jar = files("opendataloader_pdf").joinpath("jar", "opendataloader-pdf-cli.jar")
    with as_file(jar) as jar_path:
        subprocess.run(
            [
                "java",
                f"-Xmx{settings.odl_convert_max_heap}",
                # The JVM sizes GC/JIT threads by the host core count. Conversion
                # parses a single document, so more cores do not speed it up; the
                # threads just hit their limit before the heap does.
                "-XX:ActiveProcessorCount=2",
                "-Djava.awt.headless=true",
                "-Dapple.awt.UIElement=true",
                "-jar",
                str(jar_path),
                str(input_path),
                "--output-dir",
                str(output_dir),
                "--format",
                "json",
                "--quiet",
            ],
            capture_output=True,
            check=True,
            timeout=settings.odl_convert_timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
        )


def convert(pdf_bytes: bytes) -> bytes:
    """Converts a PDF to ODL JSON and returns the bytes of the result file.

    Raises:
        ConversionTimeout: The time limit was exceeded.
        ConversionFailed: java failed or no result file was produced.
    """
    work_dir = Path(tempfile.mkdtemp())
    try:
        input_path = work_dir / "input.pdf"
        input_path.write_bytes(pdf_bytes)

        output_dir = work_dir / "output"
        output_dir.mkdir()

        try:
            run_jar(input_path, output_dir)
        except subprocess.TimeoutExpired as e:
            raise ConversionTimeout("PDF conversion timed out.") from e
        except subprocess.CalledProcessError as e:
            # The exception message carries only the exit code. stderr is the only
            # signal that tells a damaged or encrypted PDF, JVM OOM, font problems and
            # flag changes after a jar upgrade apart; without logging it here,
            # operators see the same 500 for all of them.
            logging.error(
                f"[odl-convert] java conversion failed (exit {e.returncode}): "
                f"{e.stderr}"
            )
            raise ConversionFailed("PDF conversion failed.") from e

        # Collect the result
        produced = sorted(output_dir.glob("*.json"))
        if not produced:
            found = [p.name for p in output_dir.iterdir()]
            raise ConversionFailed(
                f"opendataloader-pdf result file not found. output_dir holds: {found}"
            )
        return produced[0].read_bytes()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
