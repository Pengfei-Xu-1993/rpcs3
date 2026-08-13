#!/usr/bin/env python3
"""Compile and execute the allocation-free MFC provenance table contract."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


RPCS3_DIR = pathlib.Path(__file__).resolve().parents[2]
SOURCE = pathlib.Path(__file__).with_name("ascension_mfc_provenance_table_test.cpp")


class NativeMfcProvenanceTests(unittest.TestCase):
    def test_native_table_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ascension-mfc-provenance-") as directory:
            output_dir = pathlib.Path(directory)
            executable = output_dir / (
                "ascension_mfc_provenance_table_test.exe"
                if os.name == "nt"
                else "ascension_mfc_provenance_table_test"
            )

            if os.name == "nt":
                program_files_x86 = pathlib.Path(
                    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
                )
                vswhere = program_files_x86 / "Microsoft Visual Studio/Installer/vswhere.exe"
                if not vswhere.exists():
                    self.skipTest("Visual Studio locator is unavailable")
                installation = subprocess.run(
                    [
                        str(vswhere),
                        "-latest",
                        "-products",
                        "*",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property",
                        "installationPath",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                if not installation:
                    self.skipTest("MSVC C++ build tools are unavailable")
                vcvars = pathlib.Path(installation) / "VC/Auxiliary/Build/vcvars64.bat"
                arguments = [
                    "cl.exe",
                    "/nologo",
                    "/std:c++20",
                    "/EHsc",
                    f"/I{RPCS3_DIR}",
                    str(SOURCE),
                    f"/Fe:{executable}",
                    f"/Fo:{output_dir / 'test.obj'}",
                ]
                build_script = output_dir / "compile.cmd"
                build_script.write_text(
                    "@echo off\n"
                    f'call "{vcvars}" >nul\n'
                    f"{subprocess.list2cmdline(arguments)}\n"
                    "exit /b %errorlevel%\n",
                    encoding="utf-8",
                )
                compile_process = subprocess.run(
                    ["cmd.exe", "/d", "/c", str(build_script)],
                    capture_output=True,
                    text=True,
                )
            else:
                compiler = shutil.which("c++")
                if not compiler:
                    self.skipTest("a C++20 compiler is unavailable")
                compile_process = subprocess.run(
                    [
                        compiler,
                        "-std=c++20",
                        f"-I{RPCS3_DIR}",
                        str(SOURCE),
                        "-o",
                        str(executable),
                    ],
                    capture_output=True,
                    text=True,
                )

            self.assertEqual(
                compile_process.returncode,
                0,
                compile_process.stdout + compile_process.stderr,
            )
            run_process = subprocess.run(
                [str(executable)], capture_output=True, text=True
            )
            self.assertEqual(
                run_process.returncode,
                0,
                run_process.stdout + run_process.stderr,
            )
            self.assertIn("tests passed", run_process.stdout)


if __name__ == "__main__":
    unittest.main()
