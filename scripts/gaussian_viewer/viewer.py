"""Interactive local GS-SDF Gaussian and SDF mesh browser."""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
import io
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import viser

from mesh_renderer import MeshRenderer
from renderer import Renderer, Run, interactive_view

REPO = Path(__file__).resolve().parents[2]
MAX_CLIENTS = 4
FRAME_INTERVAL = 0.1
GAUSSIAN = "Gaussian"
SDF_MESH = "SDF Mesh"
FLAT_LIGHT = "逐面光照"
SMOOTH_LIGHT = "平滑法线光照"


@dataclass
class ClientState:
    handle: viser.ClientHandle
    status: viser.GuiMarkdownHandle
    layer: str = GAUSSIAN
    smooth: bool = False
    serial: int = 0
    last_key: tuple | None = None
    latest: tuple | None = None


class Viewer:
    def __init__(self, run, renderer, initial, port, session):
        self.run = run
        self.renderer = renderer
        self.session = session
        self.initial = initial
        self.condition = threading.Condition()
        self.clients = {}
        self.pending = OrderedDict()
        self.closed = False
        self.errors = {}
        self.server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
        if self.server.get_port() != port:
            self.server.stop()
            raise RuntimeError(f"Viewer port {port} is occupied. Select another with --port.")
        self.server.scene.set_up_direction("+z")
        self.server.scene.world_axes.visible = False
        self.server.gui.add_markdown(
            "鼠标拖动旋转，右键平移，滚轮缩放。点击画布后用 WASD 移动，Q/E 升降。")
        self.server.on_client_connect(self.connect)
        self.server.on_client_disconnect(self.disconnect)
        self.thread = threading.Thread(target=self.render_loop, daemon=True)
        self.thread.start()

    def connect(self, client):
        with self.condition:
            if len(self.clients) >= MAX_CLIENTS:
                client.gui.add_markdown("已达到四个浏览连接的上限。")
                return
            status = client.gui.add_markdown("正在渲染初始视角。")
            self.clients[client.client_id] = ClientState(client, status)
        if self.run.mesh is not None:
            layer = client.gui.add_dropdown("显示", (GAUSSIAN, SDF_MESH), initial_value=GAUSSIAN)
            shading = client.gui.add_dropdown(
                "网格光照", (FLAT_LIGHT, SMOOTH_LIGHT), initial_value=FLAT_LIGHT, visible=False,
            )

            @layer.on_update
            def select_layer(_):
                shading.visible = layer.value == SDF_MESH
                self.enqueue(client, reset=True, layer=layer.value)

            shading.on_update(lambda _: self.enqueue(client, reset=True, smooth=shading.value == SMOOTH_LIGHT))
        frame = client.gui.add_text("采集帧", initial_value=self.initial)
        client.gui.add_button("跳到采集帧").on_click(lambda _: self.jump_from_gui(client, frame))
        client.gui.add_button("保存 PNG + 相机").on_click(self.save)
        self.jump(client, self.initial)
        client.camera.on_update(lambda _: self.enqueue(client))
        self.enqueue(client)

    def disconnect(self, client):
        with self.condition:
            self.clients.pop(client.client_id, None)
            self.pending.pop(client.client_id, None)

    def jump_from_gui(self, client, frame):
        try:
            name = self.run.frame_name(frame.value)
            frame.value = name
            self.jump(client, name)
        except ValueError as error:
            with self.condition:
                state = self.clients.get(client.client_id)
                if state is not None:
                    state.status.content = str(error)

    def jump(self, client, name):
        pose, camera = self.run.views[name]
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        with client.atomic():
            client.camera.position = pose[:3, 3]
            client.camera.look_at = pose[:3, 3] + pose[:3, 2] * 4
            client.camera.up_direction = -pose[:3, 1]
            client.camera.wxyz = q
            client.camera.fov = camera.fov
        self.enqueue(client, reset=True)

    def enqueue(self, client, reset=False, layer=None, smooth=None):
        try:
            view = interactive_view(client.camera.wxyz, client.camera.position,
                                    client.camera.fov, client.camera.aspect)
        except ValueError:
            return
        key = (view.pose.tobytes(), view.camera)
        client_id = client.client_id
        with self.condition:
            state = self.clients.get(client_id)
            if self.closed or state is None:
                return
            if layer is not None:
                state.layer = layer
                state.status.content = f"正在切换至 **{layer}**。"
            if smooth is not None:
                state.smooth = smooth
                lighting = SMOOTH_LIGHT if smooth else FLAT_LIGHT
                state.status.content = f"正在切换至 **{lighting}**。"
            if reset:
                # Explicit view changes invalidate an in-flight image.
                state.serial += 1
                state.latest = None
                self.pending.pop(client_id, None)
            mesh_smooth = state.layer == SDF_MESH and state.smooth
            mode = (state.layer, mesh_smooth)
            if mode in self.errors:
                state.status.content = self.errors[mode]
                return
            if not reset and key == state.last_key:
                return
            state.last_key = key
            self.pending[client_id] = (state.serial, mode, view)
            self.condition.notify()

    def render_loop(self):
        # Both backends consume the same View. The EGL context stays on this thread.
        renderers = {GAUSSIAN: self.renderer}
        try:
            self.render_pending(renderers)
        finally:
            if SDF_MESH in renderers:
                renderers[SDF_MESH].close()

    def render_pending(self, renderers):
        next_frame = 0.0
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending)
                if self.closed:
                    return
                wait = next_frame - time.monotonic()
                if wait > 0:
                    self.condition.wait(wait)
                    continue
                client_id, (serial, mode, view) = self.pending.popitem(last=False)
                if mode in self.errors:
                    continue
                layer, smooth = mode
            start = time.monotonic()
            next_frame = start + FRAME_INTERVAL
            try:
                if layer not in renderers:
                    mesh = MeshRenderer(self.run.mesh, *self.run.settings[2:4])
                    renderers[layer] = mesh
                    print(f"Loaded {mesh.count:,} mesh triangles on {mesh.device}", flush=True)
                renderer = renderers[layer]
                rgb = renderer.render(view, smooth=smooth) if layer == SDF_MESH else renderer.render(view)
            except Exception as error:
                label = SMOOTH_LIGHT if smooth else layer
                message = f"{label} 渲染失败。请查看日志。\n\n{error}"
                with self.condition:
                    self.errors[mode] = message
                    for state in self.clients.values():
                        if state.layer == layer and (layer != SDF_MESH or state.smooth == smooth):
                            state.status.content = message
                print(message, file=sys.stderr, flush=True)
                if not smooth and layer in renderers:
                    renderers.pop(layer).close()
                continue
            with self.condition:
                state = self.clients.get(client_id)
                if state is None or self.closed or serial != state.serial:
                    continue
                state.handle.scene.set_background_image(rgb, format="jpeg", jpeg_quality=95)
                state.latest = (layer, smooth, view, rgb)
                lighting = SMOOTH_LIGHT if smooth else FLAT_LIGHT
                kind = "Gaussians" if layer == GAUSSIAN else f"三角面 · {lighting}"
                state.status.content = (
                    f"**{renderer.count:,} {kind}** · {view.camera.width}×{view.camera.height}"
                    f" · 渲染 {(time.monotonic() - start) * 1000:.0f} ms"
                )

    def save(self, event):
        if event.client is None:
            return
        with self.condition:
            state = self.clients.get(event.client.client_id)
            if state is None:
                return
            if state.latest is None:
                state.status.content = "等待当前视角渲染后再保存。"
                return
            latest = state.latest
            layer, smooth, view, rgb = latest
        label = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")
        capture_dir = self.session / "captures"
        capture_dir.mkdir(exist_ok=True)
        png = io.BytesIO()
        Image.fromarray(rgb).save(png, format="PNG")
        camera = {
            "camera_to_world_opencv": view.pose.tolist(), "units": "m",
            "intrinsics": view.camera._asdict(),
            "render_layer": layer,
        }
        if layer == SDF_MESH:
            camera["mesh_shading"] = "smooth" if smooth else "flat"
        metadata = json.dumps(camera, indent=2) + "\n"
        (capture_dir / f"{label}.png").write_bytes(png.getvalue())
        (capture_dir / f"{label}.json").write_text(metadata)
        event.client.send_file_download(f"{label}.png", png.getvalue())
        event.client.send_file_download(f"{label}.json", metadata.encode())
        with self.condition:
            if state.latest is latest:
                state.status.content = f"已保存 **{label}** 的图像和相机。"

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=22)
        self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="GS-SDF training output")
    parser.add_argument("--port", type=int, default=7010)
    parser.add_argument("--frame", help="Acquisition image name or stem")
    parser.add_argument("--data-dir", type=Path, help="Override saved camera metadata directory")
    parser.add_argument("--libtorch", type=Path, help="Override CUDA LibTorch directory")
    parser.add_argument("--mesh", type=Path, help="Override exported GS-SDF mesh")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError as error:
            raise RuntimeError(f"Cannot bind 127.0.0.1:{args.port}. Select another with --port.") from error

    run = Run(args.run_dir, args.data_dir)
    if args.mesh is not None:
        run.mesh = args.mesh.resolve(strict=True)
        if not run.mesh.is_file():
            raise ValueError(f"Expected a mesh file: {run.mesh}")
    initial = run.frame_name(args.frame) if args.frame else next(iter(run.views))
    worker = (REPO / "build/gaussian-viewer/gs_sdf_viewer_worker").resolve(strict=True)
    run_root = run.path.parent.parent
    libtorch = (args.libtorch or run_root / "deps/libtorch").resolve(strict=True)
    if not (libtorch / "lib/libtorch_cuda.so").is_file():
        raise ValueError(f"Invalid CUDA LibTorch directory: {libtorch}")
    session = run_root / "viewer_sessions" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session.mkdir(parents=True)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    renderer = Renderer(run, worker, libtorch, session / "renderer.log")
    try:
        viewer = Viewer(run, renderer, initial, args.port, session)
        try:
            print(f"GS-SDF viewer: http://127.0.0.1:{args.port}/", flush=True)
            print(f"Session: {session}", flush=True)
            stopped.wait()
        finally:
            viewer.close()
    finally:
        renderer.close()


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        sys.exit(str(error))
