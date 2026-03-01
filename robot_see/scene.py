import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import singledispatch
from typing import TypedDict

import numpy as np
import PIL.Image
import trimesh
from viser import FrameHandle, ViserServer
from viser.transforms import SO3

from .model import Box, Collision, Cylinder, Inertial, Link, Mesh, PhysicalValidityWarning, Robot, Sphere, Visual


class BaseKwargs(TypedDict):
    name: str
    wxyz: tuple[float, float, float, float]
    position: tuple[float, float, float]


class ColorKwargs(TypedDict, total=False):
    color: tuple[int, int, int]
    opacity: float


class CollisionKwargs(BaseKwargs):
    color: tuple[int, int, int]
    opacity: float
    cast_shadow: bool
    receive_shadow: bool


@dataclass(frozen=True)
class SceneStyle:
    """Visual style for RobotScene"""

    collision_geom_color: tuple = (236, 236, 0)
    inertia_ellipsoid_color: tuple = (236, 0, 0)
    link_frame_color: tuple = (236, 236, 0)
    origin_frame_color: tuple = (236, 0, 236)
    principal_frame_color: tuple = (0, 236, 236)

    collision_geom_opacity: float = 0.5
    inertia_ellipsoid_opacity: float = 0.5

    # fraction of meansize
    root_frame_size: float = 1.0
    link_frame_size: float = 0.4
    origin_frame_size: float = 0.2
    principal_frame_size: float = 0.2


DEFAULT_STYLE = SceneStyle()


@singledispatch
def get_geometry_size(geometry) -> float | None:
    """Extract characteristic size from geometry"""
    return None


@get_geometry_size.register
def _(geometry: Box) -> float:
    return float(np.linalg.norm(geometry.size) / 2)


@get_geometry_size.register
def _(geometry: Sphere) -> float:
    return geometry.radius


@get_geometry_size.register
def _(geometry: Cylinder) -> float:
    return (geometry.radius + geometry.length / 2) / 2


@get_geometry_size.register
def _(geometry: Mesh) -> float | None:
    bounds = geometry.tmesh.bounds
    return float(np.linalg.norm(bounds[1] - bounds[0]) / 2) if bounds is not None else None


class RobotScene:
    """Renders a Robot model in a viser scene graph"""

    def __init__(self, server: ViserServer, robot: Robot, style: SceneStyle = DEFAULT_STYLE):
        self.server = server
        self.robot = robot
        self.style = style
        self.joint_children = defaultdict(list)

        self.link_frames = {}

        self.visual_frames = {"geometry": [], "origin": []}
        self.inertial_frames = {"geometry": [], "origin": [], "principal": []}
        self.collision_frames = {"geometry": [], "origin": []}

        self.meansize = self._compute_meansize()

        self._build_scene()

    def _compute_meansize(self) -> float:
        """Compute model size estimate for frame scaling

        Returns:
            Float estimate of model size
        """
        sizes = []

        for link in self.robot.links.values():
            for visual in link.visuals:
                visual_size = get_geometry_size(visual.geometry)
                if visual_size is not None:
                    sizes.append(visual_size)
            for collision in link.collisions:
                collision_size = get_geometry_size(collision.geometry)
                if collision_size is not None:
                    sizes.append(collision_size)
        if not sizes:
            warnings.warn("No visual or collision geometries detected", stacklevel=2)
            return 0.1
        return float(np.mean(sizes))

    def _build_scene(self) -> None:
        """Build the kinematic tree visualization in the scene graph"""
        for joint in self.robot.joints.values():
            self.joint_children[joint.parent].append(joint)

        root_link = next((link for link in self.robot.links.values() if link.parent is None), None)
        if not root_link:
            raise ValueError("No root link found")

        root_path = f"/{root_link.name}"
        root_frame = self.server.scene.add_frame(
            root_path,
            axes_length=self.style.root_frame_size * self.meansize,
            axes_radius=0.1 * self.style.root_frame_size * self.meansize,
            origin_color=self.style.link_frame_color,
            show_axes=False,
        )
        self.link_frames[root_link.name] = root_frame

        self._build_tree(root_link, root_path)

    def _build_tree(self, link: Link, link_path: str) -> None:
        """Recursively build the kinematic tree from a link node

        Args:
            link: Link node
            link_path: Scene graph path to the current link frame
        """
        self._add_visual_branches(link, link_path)
        self._add_inertial_branches(link, link_path)
        self._add_collision_branches(link, link_path)

        # recurse depth-first along joint branches
        for joint in self.joint_children[link.name]:
            child_path = f"{link_path}/{joint.child}"
            link_frame = self.server.scene.add_frame(
                name=child_path,
                wxyz=joint.origin.quat,
                position=joint.origin.xyz,
                axes_length=self.style.link_frame_size * self.meansize,
                axes_radius=0.1 * self.style.link_frame_size * self.meansize,
                origin_color=self.style.link_frame_color,
                show_axes=False,
            )
            self.link_frames[joint.child] = link_frame

            self._build_tree(self.robot.links[joint.child], child_path)

    def _add_visual_branches(self, link: Link, link_path: str) -> None:
        """Add a link's visuals to the scene graph

        Args:
            link: Link containing visuals
            link_path: Scene graph path to the link
        """
        if not link.visuals:
            return

        visual_path = f"{link_path}/visual"
        visual_frame = self.server.scene.add_frame(visual_path, show_axes=False)
        self.visual_frames["geometry"].append(visual_frame)

        for i, visual in enumerate(link.visuals):
            visual_geom_path = f"{visual_path}/vis_{i}"
            self._add_visual(visual, visual_geom_path)

    def _add_inertial_branches(self, link: Link, link_path: str) -> None:
        """Add a link's inertial information to the scene graph

        Args:
            link: Link containing inertial properties
            link_path: Scene graph path to the link
        """
        if not link.inertial:
            return

        inertial_path = f"{link_path}/inertial"
        inertial_frame = self.server.scene.add_frame(
            inertial_path,
            wxyz=link.inertial.origin.quat,
            position=link.inertial.origin.xyz,
            show_axes=False,
            visible=False,
        )
        self.inertial_frames["geometry"].append(inertial_frame)

        self._add_inertial(link.inertial, inertial_path)

    def _add_collision_branches(self, link: Link, link_path: str) -> None:
        """Add a link's collisions to the scene graph

        Args:
            link: Link containing collisions
            link_path: Scene graph path to the link frame
        """
        if not link.collisions:
            return

        collision_path = f"{link_path}/collision"
        collision_frame = self.server.scene.add_frame(collision_path, show_axes=False, visible=False)
        self.collision_frames["geometry"].append(collision_frame)

        for i, collision in enumerate(link.collisions):
            collision_geom_path = f"{collision_path}/col_{i}"
            self._add_collision(collision, collision_geom_path)

    def _add_visual(self, visual: Visual, visual_geom_path: str) -> None:
        """Add a visual geometry to the scene graph

        Args:
            visual: Visual object containing geometry
            visual_geom_path: Scene graph path to the visual geometry
        """
        color_kwargs: ColorKwargs = {}
        mat = visual.material
        if mat and mat.rgba:
            color_kwargs = {"color": tuple((np.array(mat.rgba[:3]) * 255).astype(int)), "opacity": mat.rgba[-1]}

        base_kwargs: BaseKwargs = dict(
            name=visual_geom_path,
            wxyz=visual.origin.quat,
            position=visual.origin.xyz,
        )

        match visual.geometry:
            case Box(size=s):
                self.server.scene.add_box(dimensions=s, **base_kwargs, **color_kwargs)
            case Sphere(radius=r):
                self.server.scene.add_icosphere(radius=r, **base_kwargs, **color_kwargs)
            case Cylinder(radius=r, length=h):
                self.server.scene.add_cylinder(radius=r, height=h, **base_kwargs, **color_kwargs)
            case Mesh(tmesh=m):
                if mat and mat.texture_filename and hasattr(m.visual, "uv") and m.visual.uv is not None:
                    img = PIL.Image.open(mat.texture_filename)
                    tmesh_mat = trimesh.visual.material.SimpleMaterial(image=img)
                    m.visual = trimesh.visual.texture.TextureVisuals(uv=m.visual.uv, material=tmesh_mat)
                    self.server.scene.add_mesh_trimesh(mesh=m, **base_kwargs)
                else:
                    if color_kwargs:
                        if isinstance(m, trimesh.Scene):
                            m = m.to_geometry()
                        self.server.scene.add_mesh_simple(
                            vertices=m.vertices, faces=m.faces, **base_kwargs, **color_kwargs
                        )
                    else:
                        self.server.scene.add_mesh_trimesh(mesh=m, **base_kwargs)

        self._add_origin_frame(visual_geom_path, self.visual_frames["origin"])

    def _add_inertial(self, inertial: Inertial, inertial_path: str) -> None:
        """Add an inertial ellipsoid to the scene graph

        Args:
            inertial: Inertial properties to visualize
            inertial_path: Scene graph path to the inertial frame
        """
        if inertial.mass <= 0:
            warnings.warn(
                f"\n{inertial_path} mass is non-positive: mass={inertial.mass}",
                category=PhysicalValidityWarning,
                stacklevel=2,
            )
            return

        mtx = inertial.inertia.as_matrix()
        eigvals, eigvecs = np.linalg.eigh(mtx)
        Ixx, Iyy, Izz = eigvals

        if Ixx <= 0 or Izz > Ixx + Iyy + 1e-9:
            with np.printoptions(precision=4, suppress=True):
                warnings.warn(
                    f"\n{inertial_path} inertia tensor is physically invalid:\n"
                    f"  Eigenvalues: {eigvals}\n"
                    f"  Matrix:\n{mtx}",
                    category=PhysicalValidityWarning,
                    stacklevel=2,
                )
            return

        # handle reflection
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, -1] *= -1
        rot = SO3.from_matrix(eigvecs)

        rx = np.sqrt(5 * (Iyy + Izz - Ixx) / (2 * inertial.mass))
        ry = np.sqrt(5 * (Ixx + Izz - Iyy) / (2 * inertial.mass))
        rz = np.sqrt(5 * (Ixx + Iyy - Izz) / (2 * inertial.mass))

        self.server.scene.add_icosphere(
            name=f"{inertial_path}/ellipsoid",
            color=self.style.inertia_ellipsoid_color,
            opacity=self.style.inertia_ellipsoid_opacity,
            scale=(rx, ry, rz),
            wxyz=rot.wxyz,
            cast_shadow=False,
            receive_shadow=False,
        )

        origin_frame = self.server.scene.add_frame(
            name=f"{inertial_path}/ellipsoid/origin",
            axes_length=self.style.origin_frame_size * self.meansize,
            axes_radius=0.1 * self.style.origin_frame_size * self.meansize,
            origin_color=self.style.origin_frame_color,
            wxyz=rot.inverse().wxyz,
            visible=False,
        )
        self.inertial_frames["origin"].append(origin_frame)

        principal_frame = self.server.scene.add_frame(
            name=f"{inertial_path}/ellipsoid/principal",
            axes_length=self.style.principal_frame_size * self.meansize,
            axes_radius=0.1 * self.style.principal_frame_size * self.meansize,
            origin_color=self.style.principal_frame_color,
            visible=False,
        )
        self.inertial_frames["principal"].append(principal_frame)

    def _add_collision(self, collision: Collision, collision_geom_path: str) -> None:
        """Add a collision geometry to the scene graph

        Args:
            collision: Collision object containing geometry
            collision_geom_path: Scene graph path to the collision geometry
        """
        base_kwargs: CollisionKwargs = dict(
            name=collision_geom_path,
            wxyz=collision.origin.quat,
            position=collision.origin.xyz,
            color=self.style.collision_geom_color,
            opacity=self.style.collision_geom_opacity,
            cast_shadow=False,
            receive_shadow=False,
        )

        match collision.geometry:
            case Box(size=s):
                self.server.scene.add_box(dimensions=s, **base_kwargs)
            case Sphere(radius=r):
                self.server.scene.add_icosphere(radius=r, **base_kwargs)
            case Cylinder(radius=r, length=h):
                self.server.scene.add_cylinder(radius=r, height=h, **base_kwargs)
            case Mesh(tmesh=m):
                if isinstance(m, trimesh.Scene):
                    m = m.to_geometry()
                self.server.scene.add_mesh_simple(vertices=m.vertices, faces=m.faces, **base_kwargs)

        self._add_origin_frame(collision_geom_path, self.collision_frames["origin"])

    def _add_origin_frame(self, geom_path: str, frames_list: list[FrameHandle]) -> None:
        """Add an origin frame to the scene graph

        Args:
            geom_path: Path to the parent geometry
            frames_list: List to append the frame handle to
        """
        origin_frame = self.server.scene.add_frame(
            name=f"{geom_path}/origin",
            axes_length=self.style.origin_frame_size * self.meansize,
            axes_radius=0.1 * self.style.origin_frame_size * self.meansize,
            origin_color=self.style.origin_frame_color,
            visible=False,
        )
        frames_list.append(origin_frame)

    def update_joint(self, joint_name: str, value: float) -> None:
        """Update a single joint configuration

        Args:
            joint_name: Name of the joint to update
            value: Joint position value (radians for revolute, meters for prismatic)
        """

        joint = self.robot.joints[joint_name]
        frame = self.link_frames[joint.child]

        if joint.type in ("revolute", "continuous"):
            tangent = np.array(joint.axis) * value
            rotation = SO3.exp(tangent)

            base_rotation = SO3(wxyz=np.array(joint.origin.quat))
            new_rotation = base_rotation @ rotation

            frame.wxyz = new_rotation.wxyz
            frame.position = joint.origin.xyz

        elif joint.type == "prismatic":
            translation = np.array(joint.axis) * value
            new_position = tuple(np.array(joint.origin.xyz) + translation)

            frame.wxyz = joint.origin.quat
            frame.position = new_position
