import os
from shutil import rmtree

from typing import NamedTuple


class FileInfo(NamedTuple):
    path: str
    name: str
    size: int


class LocalFs:
    mv = os.rename

    @classmethod
    def mount(cls, from_, to):
        if not os.path.exists(to):
            os.mkdir(to)

    @classmethod
    def unmount(cls, to):
        pass

    @classmethod
    def mkdirs(cls, path):
        dir_ = os.path.dirname(path)
        if not os.path.exists(dir_):
            cls.mkdirs(dir_)
        try:
            os.mkdir(path)
        except FileExistsError:
            print('The directory alreayd exists')

    @classmethod
    def ls(cls, path):
        return [FileInfo(path=os.path.join(path, p), name=p, size=0)
                for p in os.listdir(path)]

    @classmethod
    def rm(cls, path, recurse=False):
        if recurse:
            rmtree(path)
        else:
            os.remove(path)


class Secrets:
    @staticmethod
    def get(key_a: str, key_b: str):
        return ''


class LocalDbutils(NamedTuple):
    fs: LocalFs = LocalFs()
    secrets: Secrets = Secrets()
