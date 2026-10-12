"""Fetches single files out of the zipped output of a Kaggle notebook run, without downloading the whole zip."""
import io
import sys
import zipfile
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


class RemoteFile(io.RawIOBase):
    """A file on a web server, read piece by piece with range requests."""

    def __init__(self, url, block_size=4 * 1024 * 1024):
        self.url, self.block_size, self.position, self.blocks = url, block_size, 0, {}
        answer = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        answer.raise_for_status()
        self.size = int(answer.headers["Content-Range"].split("/")[1])
        self.n_bytes_fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence] + offset
        return self.position

    def block(self, index):
        if index not in self.blocks:
            start = index * self.block_size
            end = min(start + self.block_size, self.size) - 1
            answer = requests.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=300)
            answer.raise_for_status()
            self.blocks[index] = answer.content
            self.n_bytes_fetched += len(answer.content)
        return self.blocks[index]

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.position + size, self.size)
        pieces = []
        while self.position < end:
            index, offset = divmod(self.position, self.block_size)
            piece = self.block(index)[offset:offset + end - self.position]
            pieces.append(piece)
            self.position += len(piece)
        return b"".join(pieces)


kernel, target_folder, wanted_names = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
owner, slug = kernel.split("/")
api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as client:
    request = ApiListKernelSessionOutputRequest()
    request.user_name, request.kernel_slug = owner, slug
    files = client.kernels.kernels_api_client.list_kernel_session_output(request).files
archive_url = next(item.url for item in files if item.file_name == "_output_.zip")
remote = RemoteFile(archive_url)
print(f"output archive: {remote.size / 1e9:.2f} GB")
with zipfile.ZipFile(remote) as archive:
    names = archive.namelist()
    print("files in the archive:", len(names))
    for wanted in wanted_names:
        member = archive.getinfo(wanted)
        (target_folder / Path(wanted).name).write_bytes(archive.read(member))
        print(f"saved {Path(wanted).name}: {member.file_size} bytes, modified {member.date_time}")
print(f"downloaded {remote.n_bytes_fetched / 1e6:.0f} MB in total")
