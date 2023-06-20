import os
import json
import tarfile
from functools import partial
from datasets.utils.py_utils import NestedDataStructure

# https://github.com/huggingface/datasets/pull/3815
# from datasets.utils.download_manager import ArchiveIterable
from datasets.utils.file_utils import (
    cached_path,
    is_relative_path,
    url_or_path_join,
    DownloadConfig,
)


data_url = "http://klue-benchmark.com.s3.amazonaws.com/app/Competitions/000073/data/wos-v1.tar.gz"


def iter_archive(path: str):
    def _iter_archive(f):
        stream = tarfile.open(fileobj=f, mode="r|*")
        for tarinfo in stream:
            file_path = tarinfo.name
            if not tarinfo.isreg():
                continue
            if file_path is None:
                continue
            if (
                os.path.basename(file_path).startswith(".")
                or os.path.basename(file_path).startswith("__")
            ):
                # skipping hidden files
                continue
            file_obj = stream.extractfile(tarinfo)
            yield file_path, file_obj
            stream.members = []
        del stream

    with open(path, "rb") as f:
        yield from _iter_archive(f)


def _download(url_or_filename: str, download_config: DownloadConfig) -> str:
    url_or_filename = str(url_or_filename)
    if is_relative_path(url_or_filename):
        # append the relative path to the base_path
        url_or_filename = url_or_path_join(os.path.abspath("."), url_or_filename)
    return cached_path(url_or_filename, download_config=download_config)


def load_json_file(filename="ontology.json", **kwargs):
    dl_config = DownloadConfig(**kwargs)

    download_fn = partial(_download, download_config=dl_config)
    archive = download_fn(data_url)

    dir_name = data_url.split("/")[-1].replace(".tar.gz", "")

    # files = ArchiveIterable.from_path(archive)
    for path, f in iter_archive(archive):
        if path == dir_name + "/" + filename:
            file = json.load(f)

    return file
