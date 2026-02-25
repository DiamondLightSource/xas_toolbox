import subprocess
import sys

from xas_toolbox import __version__


def test_cli_version():
    cmd = [sys.executable, "-m", "xas_toolbox", "--version"]
    assert subprocess.check_output(cmd).decode().strip() == __version__
