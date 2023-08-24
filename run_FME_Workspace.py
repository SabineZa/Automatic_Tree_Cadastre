# -*- coding: utf-8 -*-
"""
Automatic Tree Cadastre
This program automatically creates a tree cadastre from a point cloud.

Copyright (c) 2022-2023 Sabine Zagst (s.zagst@tum.de)

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from pathlib import Path
import subprocess


def run_FME_Workspace() -> None:
    """Run a FME Workspace from Python"""

    if not Path(r"C:\Program Files\FME\fme.exe").exists():
        print("FME is not installed, aborting...")
        return

    args = [
        r"C:\Program Files\FME\fme.exe",
        "csvfile2citygml.fmw",
    ]
    subprocess.Popen(args)
