from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

__all__ = [
    "Robot",
    "Link",
    "Joint",
    "Limit",
    "Visual",
    "Material",
    "Collision",
    "Mesh",
    "Sphere",
    "Cylinder",
    "Box",
    "Geometry",
    "Inertial",
    "Inertia",
    "Pose",
    "PhysicalValidityWarning",
    "Base",
]


class PhysicalValidityWarning(UserWarning):
    pass


@dataclass
class Base:
    """Base class for objects with source tracking metadata

    Attributes:
        line_number: Line number of element in source file
        source_path: Hierarchical path to element in source format
        source_file: Path to source file
    """

    _line_number: int | None = field(default=None, repr=False, compare=False, kw_only=True)
    _source_path: str | None = field(default=None, repr=False, compare=False, kw_only=True)
    _source_file: str | None = field(default=None, repr=False, compare=False, kw_only=True)


@dataclass
class Pose(Base):
    """Position and orientation in SE(3)

    Attributes:
        xyz: (x, y, z) position in meters, defaults to (0.0, 0.0, 0.0)
        quat: (w, x, y, z) unit quaternion in radians, defaults to (1.0, 0.0, 0.0, 0.0)
    """

    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class Inertia(Base):
    """Inertia tensor

    Attributes:
        ixx, ixy, ixz, iyy, iyz, izz: Components of the 3x3 symmetric inertia tensor in kg*m^2
    """

    ixx: float = 0.0
    ixy: float = 0.0
    ixz: float = 0.0
    iyy: float = 0.0
    iyz: float = 0.0
    izz: float = 0.0

    def as_matrix(self):
        """Return the 3x3 symmetric inertia tensor matrix."""
        return np.array(
            [[self.ixx, self.ixy, self.ixz], [self.ixy, self.iyy, self.iyz], [self.ixz, self.iyz, self.izz]]
        )


@dataclass
class Inertial(Base):
    """Inertial properties of a link

    Attributes:
        origin: Pose of inertial frame w.r.t. link frame, defaults to the identity
        mass: Mass in kilograms
        inertia: Inertia tensor
    """

    origin: Pose = field(default_factory=Pose)
    mass: float = 0.0
    inertia: Inertia = field(default_factory=Inertia)


@dataclass
class Geometry(Base):
    """Base class for geometric shapes"""

    pass


@dataclass
class Box(Geometry):
    """Box geometry

    Attributes:
        size: (x, y, z) dimensions in meters
    """

    size: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Cylinder(Geometry):
    """Cylinder geometry

    Attributes:
        radius: Radius in meters
        length: Length in meters
    """

    radius: float = 0.0
    length: float = 0.0


@dataclass
class Sphere(Geometry):
    """Sphere geometry

    Attributes:
        radius: Radius in meters
    """

    radius: float = 0.0


@dataclass
class Mesh(Geometry):
    """Mesh geometry

    Attributes:
        filename: Path to mesh file
        scale: (x, y, z) scaling factors, defaults to (1.0, 1.0, 1.0)
    """

    filename: Path | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    _tmesh: trimesh.Trimesh | trimesh.Scene | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def tmesh(self):
        """Get trimesh object"""
        if self._tmesh is None and self.filename:
            ext = self.filename.suffix.lower()
            if ext in (".obj", ".stl"):
                mesh = trimesh.load_mesh(self.filename)
            else:
                mesh = trimesh.load_scene(self.filename)
            mesh.apply_scale(self.scale)
            self._tmesh = mesh
        return self._tmesh

    @tmesh.setter
    def tmesh(self, mesh: trimesh.Trimesh | trimesh.Scene):
        """Directly assign a trimesh object"""
        self._tmesh = mesh


@dataclass
class Collision(Base):
    """Collision geometry of a link

    Attributes:
        name: Name of the collision element, defaults to None if not specified
        origin: Pose of collision geometry w.r.t. link frame, defaults to the identity
        geometry: Geometric shape for collision checking
    """

    name: str | None = None
    origin: Pose = field(default_factory=Pose)
    geometry: Geometry | None = None


@dataclass
class Material(Base):
    """Material properties of a visual element

    Attributes:
        name: Name of the material, defaults to None if not specified
        rgba: (r, g, b, a) color values from 0-1
        texture_filename: Path to texture file
    """

    name: str | None = None
    rgba: tuple[float, float, float, float] | None = None
    texture_filename: Path | None = None


@dataclass
class Visual(Base):
    """Visual geometry of a link

    Attributes:
        origin: Pose of visual geometry w.r.t. link frame, defaults to the identity
        geometry: Geometric shape for visualization
        material: Material properties
    """

    origin: Pose = field(default_factory=Pose)
    geometry: Geometry | None = None
    material: Material | None = None


@dataclass
class Limit(Base):
    """Joint limits

    Attributes:
        lower: Lower joint limit (radians for revolute, meters for prismatic)
        upper: Upper joint limit (radians for revolute, meters for prismatic)
        effort: Maximum joint effort (torque for revolute, force for prismatic)
        velocity: Maximum joint velocity (rad/s for revolute, m/s for prismatic)
    """

    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0


@dataclass
class Joint(Base):
    """Joint connecting two links

    Attributes:
        name: Name of the joint
        type: Type of joint (e.g. 'revolute')
        parent: Name of the parent link
        child: Name of the child link
        origin: Pose of child link frame w.r.t. parent link frame, defaults to the identity
        axis: (x, y, z) axis of actuation expressed in the child link frame, defaults to (1.0, 0.0, 0.0)
    """

    name: str
    type: str
    parent: str
    child: str
    origin: Pose = field(default_factory=Pose)
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    limit: Limit | None = None


@dataclass
class Link(Base):
    """Robot link

    Attributes:
        name: Name of the link
        inertial: Inertial properties
        collisions: List of Collision objects
        visuals: List of Visual objects
    """

    name: str
    parent: str | None = None
    inertial: Inertial | None = None
    collisions: list[Collision] = field(default_factory=list)
    visuals: list[Visual] = field(default_factory=list)


@dataclass
class Robot:
    """Abstract robot representation

    Attributes:
        name: Name of the robot
        links: Dict mapping link names to Link objects
        joints: Dict mapping joint names to Joint objects
    """

    name: str
    links: dict[str, Link] = field(default_factory=dict)
    joints: dict[str, Joint] = field(default_factory=dict)
