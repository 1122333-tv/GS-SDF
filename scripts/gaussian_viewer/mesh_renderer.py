"""Offscreen rendering of a GS-SDF triangle mesh with the viewer's camera."""

import numpy as np


VERTEX_SHADER = """
#version 330
in vec3 position;
in vec3 vertex_normal;
uniform mat4 world_to_camera;
uniform mat4 camera_to_clip;
uniform bool smooth_shading;
out vec3 camera_position;
out vec3 camera_normal;
void main() {
    vec4 camera_point = world_to_camera * vec4(position, 1.0);
    camera_position = camera_point.xyz;
    camera_normal = smooth_shading ? mat3(world_to_camera) * vertex_normal : vec3(0.0);
    gl_Position = camera_to_clip * camera_point;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 camera_position;
in vec3 camera_normal;
uniform bool smooth_shading;
out vec4 color;
void main() {
    // Face normals come from the triangle's interpolated camera-space position.
    vec3 face_normal = cross(dFdx(camera_position), dFdy(camera_position));
    vec3 normal = normalize(smooth_shading && dot(camera_normal, camera_normal) > 1e-12
                            ? camera_normal : face_normal);
    if (dot(normal, camera_position) > 0.0) normal = -normal;
    float key = max(dot(normal, normalize(vec3(-0.55, -0.70, -0.60))), 0.0);
    float fill = max(dot(normal, normalize(vec3(0.85, 0.12, -0.52))), 0.0);
    vec3 surface = vec3(0.78, 0.81, 0.86) * (0.31 + 0.68 * key + 0.22 * fill);
    vec3 background = vec3(0.09, 0.12, 0.16);
    float haze = min(1.0 - exp(-0.023 * camera_position.z), 0.65);
    color = vec4(mix(surface, background, haze), 1.0);
}
"""

VERTEX = np.dtype([("position", "<f4", (3,)), ("rgb", "u1", (3,))])
TRIANGLE = np.dtype([("count", "<i4"), ("indices", "<u4", (3,))])


def read_mesh(path):
    # GS-SDF writes packed binary vertices and a fixed-size triangle list.
    with path.open("rb") as file:
        header = []
        while len(header) < 128:
            line = file.readline(1024).decode("ascii").strip()
            if not line or line == "end_header":
                break
            if not line.startswith("comment "):
                header.append(line)
        offset = file.tell()
    expected_properties = [
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
    ]
    if (line != "end_header" or len(header) != 11 or
            header[:2] != ["ply", "format binary_little_endian 1.0"] or
            not header[2].startswith("element vertex ") or header[3:9] != expected_properties or
            not header[9].startswith("element face ") or
            header[10] != "property list int int vertex_index"):
        raise ValueError(f"Expected a binary GS-SDF triangle mesh: {path}")
    vertex_count = int(header[2].split()[-1])
    face_count = int(header[9].split()[-1])
    face_offset = offset + vertex_count * VERTEX.itemsize
    if (min(vertex_count, face_count) <= 0 or
            path.stat().st_size != face_offset + face_count * TRIANGLE.itemsize):
        raise ValueError(f"Mesh has inconsistent counts or file size: {path}")
    vertices = np.memmap(path, mode="r", dtype=VERTEX, offset=offset, shape=(vertex_count,))
    faces = np.memmap(path, mode="r", dtype=TRIANGLE, offset=face_offset, shape=(face_count,))
    if not np.all(faces["count"] == 3) or faces["indices"].max() >= vertex_count:
        raise ValueError(f"Invalid triangle indices in {path}")
    return vertices["position"], faces["indices"]


def camera_to_clip(camera, near, far):
    width, height, fx, fy, cx, cy = camera
    # OpenCV's right/down/forward axes project to OpenGL's up-facing viewport.
    return np.array([
        [2 * fx / width, 0, 2 * cx / width - 1, 0],
        [0, -2 * fy / height, 1 - 2 * cy / height, 0],
        [0, 0, (far + near) / (far - near), -2 * near * far / (far - near)],
        [0, 0, 1, 0],
    ], dtype="f4")


def vertex_normals(vertices, faces):
    """Average area-weighted face normals across exactly coincident vertices."""
    chunk_size = 500_000
    used = np.zeros(len(vertices), dtype=bool)
    for start in range(0, len(faces), chunk_size):
        used[faces[start:start + chunk_size]] = True
    active = np.flatnonzero(used)
    positions = np.ascontiguousarray(vertices[active])
    positions[positions == 0] = 0  # Treat positive and negative zero alike.
    keys = positions.view(np.dtype((np.void, positions.dtype.itemsize * 3))).ravel()
    unique, inverse = np.unique(keys, return_inverse=True)
    shared = np.empty(len(vertices), dtype=np.int32)
    shared[active] = inverse
    accumulated = np.zeros((len(unique), 3), dtype=np.float32)
    del positions, keys, unique, inverse
    for start in range(0, len(faces), chunk_size):
        face = faces[start:start + chunk_size]
        points = vertices[face]
        normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        for corner in face.T:
            np.add.at(accumulated, shared[corner], normals)
    lengths = np.linalg.norm(accumulated, axis=1, keepdims=True)
    np.divide(accumulated, lengths, out=accumulated, where=lengths > 0)
    result = np.zeros((len(vertices), 3), dtype=np.float32)
    for start in range(0, len(active), chunk_size):
        indices = active[start:start + chunk_size]
        result[indices] = accumulated[shared[indices]]
    return result


class MeshRenderer:
    def __init__(self, path, near, far):
        import moderngl

        vertices, faces = read_mesh(path)
        self.source = vertices, faces
        self.count = len(faces)
        self.near, self.far = near, far
        self.context = moderngl.create_standalone_context(require=330, backend="egl")
        self.device = self.context.info["GL_RENDERER"]
        self.framebuffer = None
        self.smooth_mesh = None
        try:
            self.program = self.context.program(vertex_shader=VERTEX_SHADER,
                                                fragment_shader=FRAGMENT_SHADER)
            self.vertices = self.context.buffer(np.ascontiguousarray(vertices))
            self.indices = self.context.buffer(np.ascontiguousarray(faces))
            self.mesh = self.context.vertex_array(
                self.program, [(self.vertices, "3f", "position")],
                index_buffer=self.indices, index_element_size=4,
            )
            self.context.enable(moderngl.DEPTH_TEST)
        except Exception:
            self.close()
            raise

    def release_framebuffer(self):
        if self.framebuffer is not None:
            attachments = (*self.framebuffer.color_attachments, self.framebuffer.depth_attachment)
            self.framebuffer.release()
            for attachment in attachments:
                attachment.release()
            self.framebuffer = None

    def render(self, view, smooth=False):
        if smooth and self.smooth_mesh is None:
            normals = self.context.buffer(vertex_normals(*self.source))
            try:
                self.smooth_mesh = self.context.vertex_array(
                    self.program, [(self.vertices, "3f", "position"), (normals, "3f", "vertex_normal")],
                    index_buffer=self.indices, index_element_size=4,
                )
            except Exception:
                normals.release()
                raise
            self.normal_buffer = normals
            self.source = None
        size = view.camera.width, view.camera.height
        if self.framebuffer is None or self.framebuffer.size != size:
            self.release_framebuffer()
            self.framebuffer = self.context.simple_framebuffer(size, components=3)
        self.framebuffer.use()
        self.framebuffer.clear(0.09, 0.12, 0.16, depth=1.0)
        rotation = view.pose[:3, :3].T
        world_to_camera = np.eye(4, dtype="f4")
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = -rotation @ view.pose[:3, 3]
        self.program["world_to_camera"].write(world_to_camera.T.tobytes())
        self.program["camera_to_clip"].write(camera_to_clip(view.camera, self.near, self.far).T.tobytes())
        self.program["smooth_shading"].value = smooth
        mesh = self.smooth_mesh if smooth else self.mesh
        mesh.render()
        pixels = self.framebuffer.read(components=3, alignment=1)
        return np.frombuffer(pixels, dtype=np.uint8).reshape(size[1], size[0], 3)[::-1].copy()

    def close(self):
        self.release_framebuffer()
        self.context.release()
