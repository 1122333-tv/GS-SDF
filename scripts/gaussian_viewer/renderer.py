"""Load GS-SDF training metadata and communicate with its native renderer."""

import os
from pathlib import Path
import select
import subprocess
import time
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


class Camera(NamedTuple):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def fov(self):
        return 2 * np.arctan(self.height / (2 * self.fy))


class View(NamedTuple):
    pose: np.ndarray
    camera: Camera

    def request(self):
        values = (*self.camera, *self.pose[:3, :4].ravel())
        return ("RENDER " + " ".join(str(value) for value in values) + "\n").encode()


def load_yaml(path):
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("%YAML:"):
        lines.pop(0)
    return yaml.safe_load("\n".join(lines))


def read_cameras(path, scale):
    cameras = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 8 or fields[1] != "PINHOLE":
            raise ValueError(f"Expected an undistorted PINHOLE camera in {path}")
        camera_id, _, width, height, *intrinsics = fields
        camera = Camera(int(int(width) * scale), int(int(height) * scale),
                        *(float(v) * scale for v in intrinsics))
        if (not np.isfinite(camera).all() or min(camera.width, camera.height) <= 0
                or min(camera.fx, camera.fy) <= 0):
            raise ValueError(f"Invalid camera {camera_id} in {path}")
        cameras[camera_id] = camera
    return cameras


def read_views(path, cameras):
    lines = iter(path.read_text().splitlines())
    views = {}
    for line in lines:
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 10 or fields[8] not in cameras or next(lines, None) is None:
            raise ValueError(f"Invalid two-line COLMAP pose in {path}: {line}")
        _, qw, qx, qy, qz, tx, ty, tz, camera_id, name = fields
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat([float(q) for q in (qx, qy, qz, qw)]).as_matrix().T
        pose[:3, 3] = -pose[:3, :3] @ np.array([float(t) for t in (tx, ty, tz)])
        if not np.isfinite(pose).all() or name in views:
            raise ValueError(f"Invalid or duplicate camera pose: {name}")
        views[name] = View(pose, cameras[camera_id])
    if not views:
        raise ValueError(f"No camera poses in {path}")
    return views


class Run:
    def __init__(self, path, data_dir=None):
        self.path = path.resolve(strict=True)
        self.model = (self.path / "model/gs.ply").resolve(strict=True)
        cfg_path = self.path / "model/config/scene/config.yaml"
        cfg = load_yaml(cfg_path)
        scene = load_yaml(cfg_path.parent / cfg["scene_config"]) if "scene_config" in cfg else cfg
        base = load_yaml(cfg_path.parent / cfg["base_config"])
        data = (data_dir or Path(cfg["data_path"])).resolve(strict=True)
        if int(scene["color_pose_type"]) != 4 or not int(scene["color_pose_w2c"]):
            raise ValueError("Viewer requires COLMAP world-to-camera image poses")
        cameras = read_cameras(data / scene["camera_path"], float(cfg.get("camera", {}).get("scale", 1)))
        self.views = read_views(data / scene["color_pose_path"], cameras)
        resolution = cfg.get("export_resolution")
        self.mesh = self.path / f"gs_log/mesh_{float(resolution):.6f}.ply" if resolution else None
        if self.mesh is not None and not self.mesh.is_file():
            self.mesh = None
        self.settings = (
            int(scene["sh_degree"]), int(scene["bck_color"]), float(base["near"]),
            float(base["far"]), int(base.get("use_absgrad", 0)), int(base.get("center_reg", 0)),
        )

    def frame_name(self, value):
        name = value.strip()
        if name in self.views:
            return name
        matches = [candidate for candidate in self.views if Path(candidate).stem == name]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Unknown or ambiguous acquisition frame: {value}")


def interactive_view(wxyz, position, fov, aspect, long_edge=768):
    if not np.isfinite([fov, aspect]).all() or not (0.02 < fov < 3.12 and 0.05 <= aspect <= 20):
        raise ValueError("Invalid camera field of view or aspect ratio")
    width, height = ((long_edge, round(long_edge / aspect)) if aspect >= 1
                     else (round(long_edge * aspect), long_edge))
    fy = height / (2 * np.tan(fov / 2))
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix()
    pose[:3, 3] = position
    if not np.isfinite(pose).all():
        raise ValueError("Invalid camera pose")
    return View(pose, Camera(width, height, fy, fy, width / 2, height / 2))


class Renderer:
    def __init__(self, run, worker, libtorch, log_path):
        self.name = f"gs-sdf-viewer-{os.getpid()}"
        self.log_path = log_path
        self.buffer = bytearray()
        self.process = None
        command = [
            "docker", "run", "--rm", "--init", "-i", "--pull=never", "--network", "none",
            "--read-only", "--gpus", "device=0", "--memory", "8g", "--cpus", "2",
            "--user", f"{os.getuid()}:{os.getgid()}", "--name", self.name,
            "--env", "CUDA_CACHE_DISABLE=1", "--env", "OMP_NUM_THREADS=2",
            "--mount", f"type=bind,source={worker},target=/opt/gs-sdf-viewer-worker,readonly",
            "--mount", f"type=bind,source={libtorch},target=/opt/viewer-libtorch,readonly",
            "--mount", f"type=bind,source={run.model},target=/scene/gs.ply,readonly",
            "--entrypoint", "/opt/gs-sdf-viewer-worker",
            os.environ.get("GS_SDF_CUDA_IMAGE", "gs-sdf-cu013:cu118-jammy"),
            "/scene/gs.ply", *map(str, run.settings),
        ]
        with log_path.open("wb") as log:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=log, bufsize=0)
        try:
            ready = self.read_header(time.monotonic() + 90)
            if len(ready) != 2 or ready[0] != b"READY" or int(ready[1]) <= 0:
                raise RuntimeError(f"Invalid renderer handshake: {ready!r}")
            self.count = int(ready[1])
        except Exception:
            self.close()
            raise

    def read_more(self, deadline):
        fd = self.process.stdout.fileno()
        if not select.select([fd], [], [], max(0.0, deadline - time.monotonic()))[0]:
            raise TimeoutError(f"GS-SDF renderer timed out. See {self.log_path}")
        data = os.read(fd, 65536)
        if not data:
            raise RuntimeError(f"GS-SDF renderer disconnected. See {self.log_path}")
        self.buffer.extend(data)

    def read_header(self, deadline):
        while b"\n" not in self.buffer:
            if len(self.buffer) > 256:
                raise RuntimeError("Invalid renderer protocol header")
            self.read_more(deadline)
        end = self.buffer.index(b"\n")
        if end > 256:
            raise RuntimeError("Invalid renderer protocol header")
        header = bytes(self.buffer[:end]).split()
        del self.buffer[:end + 1]
        return header

    def render(self, view):
        self.process.stdin.write(view.request())
        deadline = time.monotonic() + 20
        width, height = view.camera[:2]
        size = width * height * 3
        if self.read_header(deadline) != f"FRAME {width} {height} {size}".encode().split():
            raise RuntimeError("Unexpected renderer frame shape")
        while len(self.buffer) < size:
            self.read_more(deadline)
        rgb = np.frombuffer(bytes(self.buffer[:size]), dtype=np.uint8).reshape(height, width, 3)
        del self.buffer[:size]
        return rgb

    def close(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(["docker", "stop", "-t", "4", self.name], timeout=8,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                pass
            if process.poll() is None:
                process.kill()
                process.wait()
        process.stdout.close()
