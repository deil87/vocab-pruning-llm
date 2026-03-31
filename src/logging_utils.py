"""
Tee stdout+stderr to a log file while keeping live tqdm progress in terminal.

Usage:
    from src.logging_utils import setup_logging
    setup_logging("results/run_foo.log")

All print() calls and tqdm bars are written to both the terminal and the log
file.  In the log file tqdm progress lines are written once per completed bar
(no carriage-return spam).
"""
import io
import sys
import time
from pathlib import Path


class _TeeStream(io.TextIOBase):
    """
    Wraps an underlying terminal stream and mirrors output to a log file.

    Rules:
      - Characters that arrive before a newline are buffered.
      - A '\\r' (tqdm in-place update) flushes the current terminal line but
        only writes to the log file if the buffer ends with a newline first,
        i.e. it's treated as a line-overwrite and NOT logged mid-progress.
      - A '\\n' flushes both terminal and log file.
    """

    def __init__(self, terminal_stream, log_file):
        super().__init__()
        self._term = terminal_stream
        self._log = log_file
        self._buf = ""          # accumulates chars until \n
        self._last_cr_line = "" # last line received via \r (for log on completion)

    # ── TextIOBase interface ──────────────────────────────────────────────────

    @property
    def encoding(self):
        return getattr(self._term, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._term, "errors", "replace")

    def readable(self):   return False
    def writable(self):   return True
    def seekable(self):   return False

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._term.write(text)
        self._term.flush()

        # Mirror to log file, collapsing \r updates to single lines
        for ch in text:
            if ch == "\r":
                # tqdm in-place update: remember last content but don't log yet
                self._last_cr_line = self._buf
                self._buf = ""
            elif ch == "\n":
                # Completed line — if there was a \r pending, prefer that
                line = self._last_cr_line or self._buf
                self._log.write(line + "\n")
                self._log.flush()
                self._buf = ""
                self._last_cr_line = ""
            else:
                self._buf += ch

        return len(text)

    def flush(self):
        self._term.flush()
        self._log.flush()

    def fileno(self):
        # Some libraries (e.g. tqdm isatty check) call fileno; delegate.
        return self._term.fileno()

    def isatty(self):
        return self._term.isatty()


def setup_logging(log_path: str | Path) -> Path:
    """
    Redirect sys.stdout and sys.stderr through a Tee that also writes to
    *log_path*.  Returns the resolved Path of the log file.

    Call once at the top of each run_*.py main() function:

        from src.logging_utils import setup_logging
        setup_logging(CFG.results_dir / "run_dual_encoder.log")
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    # Write a header so restarts are visible in the log
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"\n{'='*60}\n  Run started: {ts}\n{'='*60}\n")
    log_file.flush()

    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)

    return log_path
