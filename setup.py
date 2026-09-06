from setuptools import setup
from email.parser import Parser
from pathlib import Path
import re


def derive_version() -> str:
    version = ''
    with open('discord/__init__.py') as f:
        version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

    if not version:
        raise RuntimeError('version is not set')

    if version.endswith(('a', 'b', 'rc')):
        # Source archives have no Git history; preserve the version recorded
        # when the sdist was built instead of falling back to an alpha of zero.
        metadata_path = Path('PKG-INFO')
        if metadata_path.is_file():
            metadata = Parser().parsestr(metadata_path.read_text(encoding='utf-8'))
            if metadata['Version']:
                return metadata['Version']

        # append version identifier based on commit count
        try:
            import subprocess

            p = subprocess.Popen(['git', 'rev-list', '--count', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            if out:
                version += out.decode('utf-8').strip()
            p = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            if out:
                version += '+g' + out.decode('utf-8').strip()
        except Exception:
            pass

    return version


setup(version=derive_version())
