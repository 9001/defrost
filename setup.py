#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function, unicode_literals

import os
import sys
import shutil
import subprocess as sp
from glob import glob
from setuptools import setup, Command


class ST_cmd(Command):
    description = "foo"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class cln(ST_cmd):
    def run(self):
        for d in ["build", "dist", "defrostir.egg-info"]:
            try:
                shutil.rmtree(d)
            except:
                pass


class tst(ST_cmd):
    def run(self):
        do_rls(False)


class rls(ST_cmd):
    def run(self):
        do_rls(True)


def sh(bin, *args, **kwargs):
    cmd = [bin] + " ".join(args).split(" ")
    print(f"\n\033[1;37;44m{repr(cmd)}\033[0m")
    sp.check_call(cmd, **kwargs)


def do_rls(for_real):
    env = os.environ.copy()
    for ek, tk in [["u", "TWINE_USERNAME"], ["p", "TWINE_PASSWORD"]]:
        v = os.environ.get(ek, "")
        if v:
            env[tk] = v

    py = sys.executable
    sp.run("rem", shell=True)
    try:
        import twine, wheel, setuptools
    except:
        sh(py, "-m pip install --user twine wheel setuptools")

    sh(py, "setup.py cln")
    sh(py, "setup.py sdist bdist_wheel --universal")
    sh(py, "setup.py sdist bdist_wheel --universal --universal")

    files = glob(os.path.join("dist", "*"))
    dest = "pypi" if for_real else "testpypi"
    sh(py, "-m twine upload -r", dest, *files, env=env)


if not os.path.isdir("defrostir"):
    shutil.copytree("defrost", "defrostir", symlinks=True)

    def uncopy():
        shutil.rmtree("defrostir")

    import atexit

    atexit.register(uncopy)


with open("README.md", encoding="utf8") as f:
    readme = f.read()

a = {}
with open("defrostir/__main__.py", encoding="utf8") as f:
    exec(f.read().split("\nimport", 1)[0], a)

a = a["about"]
del a["date"]

if sys.argv[-2:] == ["--universal", "--universal"]:
    chardet = "charset_normalizer"
    a["version"] += ".2"
elif sys.argv[-2:] == ["bdist_wheel", "--universal"]:
    chardet = "chardet"
    a["version"] += ".1"
else:
    chardet = "charset_normalizer" if sys.version_info[0] > 2 else "chardet"

a.update(
    {
        "author_email": "@".join([a["name"], "ocv.me"]),
        "python_requires": ">=2.7",
        "install_requires": ["mutagen", chardet],
        "entry_points": {"console_scripts": ["defrostir=defrostir.__main__:main"]},
        "include_package_data": True,
        "long_description": readme,
        "long_description_content_type": "text/markdown",
        "keywords": "icecast internet radio stream ripping silence splitting",
        "classifiers": [
            "License :: OSI Approved :: MIT License",
            "Development Status :: 5 - Production/Stable",
            "Environment :: Console",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS",
            "Topic :: Multimedia :: Sound/Audio",
            "Intended Audience :: End Users/Desktop",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
        ],
        "packages": [a["name"]],
        "cmdclass": {"cln": cln, "rls": rls, "tst": tst},
    }
)

setup(**a)
