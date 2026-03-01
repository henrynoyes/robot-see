from collections.abc import Callable
from dataclasses import dataclass, field

from viser import FrameHandle, ViserServer

from .model import Joint

# joint sliders
_JOINT_SLIDER_STEP = 0.01
_JOINT_DEFAULT_LOWER = -3.14
_JOINT_DEFAULT_UPPER = 3.14

# frame scale sliders
_FRAME_SCALE_MIN = 0.2
_FRAME_SCALE_MAX = 5.0
_FRAME_SCALE_STEP = 0.1


@dataclass
class VisibilityState:
    geometry: bool = True
    origin: bool = False
    principal: bool = False


@dataclass
class GUIState:
    link_frames_visible: bool = False
    visual: VisibilityState = field(default_factory=VisibilityState)
    collision: VisibilityState = field(default_factory=lambda: VisibilityState(geometry=False))
    inertial: VisibilityState = field(default_factory=lambda: VisibilityState(geometry=False))
    joint_positions: dict[str, float] = field(default_factory=dict)
    frame_scales: dict[str, float] = field(default_factory=dict)


def add_visibility_folder(
    server: ViserServer,
    label: str,
    frames: dict[str, list[FrameHandle]],
    state: VisibilityState,
) -> None:
    """Add GUI checkboxes to toggle frame group visibility

    Args:
        server: Viser server
        label: Label for the group
        frames: Dict of frame lists within the group
        state: VisibilityState to synchronize
    """
    with server.gui.add_folder(label):
        geometry_checkbox = server.gui.add_checkbox("Geometry", initial_value=state.geometry)

        origin_checkbox = server.gui.add_checkbox(
            "Origin Frames", initial_value=state.origin, disabled=not state.geometry
        )

        @geometry_checkbox.on_update
        def _(event):
            state.geometry = event.target.value
            for frame in frames["geometry"]:
                frame.visible = event.target.value
            origin_checkbox.disabled = not event.target.value
            if principal_checkbox:
                principal_checkbox.disabled = not event.target.value

        @origin_checkbox.on_update
        def _(event):
            state.origin = event.target.value
            for frame in frames["origin"]:
                frame.visible = event.target.value

        principal_checkbox = None
        if frames.get("principal"):
            principal_checkbox = server.gui.add_checkbox(
                "Principal Frames", initial_value=state.principal, disabled=not state.geometry
            )

            @principal_checkbox.on_update
            def _(event):
                state.principal = event.target.value
                for frame in frames["principal"]:
                    frame.visible = event.target.value

            principal_checkbox.value = principal_checkbox.value

        geometry_checkbox.value = geometry_checkbox.value
        origin_checkbox.value = origin_checkbox.value


def add_joint_sliders(
    server: ViserServer,
    joints: dict[str, Joint],
    on_change: Callable[[str, float], None],
    state: dict[str, float],
):
    """Add GUI sliders for position control of actuated joints

    Args:
        server: Viser server
        joints: Dict mapping joint names to Joint objects
        on_change: Callback function (joint_name, value) for joint update
        state: Dict mapping joint names to positions
    """
    slider_data = []

    with server.gui.add_folder("Joint Position Control", expand_by_default=False):
        for name, joint in joints.items():
            if joint.type not in ("revolute", "continuous", "prismatic"):
                continue

            if joint.limit:
                lower, upper = joint.limit.lower, joint.limit.upper
            else:
                lower, upper = _JOINT_DEFAULT_LOWER, _JOINT_DEFAULT_UPPER

            initial = state.get(name)
            if initial is not None:
                initial = max(lower, min(upper, initial))
            else:
                initial = 0.0 if lower <= 0.0 <= upper else lower

            slider = server.gui.add_slider(
                name,
                min=lower,
                max=upper,
                step=_JOINT_SLIDER_STEP,
                initial_value=initial,
            )
            slider_data.append((slider, initial))

            @slider.on_update
            def _(event, joint_name=name):
                state[joint_name] = event.target.value
                on_change(joint_name, event.target.value)

            slider.value = slider.value

        reset_button = server.gui.add_button("Reset Joints")

        @reset_button.on_click
        def _(_):
            for slider, initial_value in slider_data:
                slider.value = initial_value
            state.clear()


def add_frame_scale_sliders(
    server: ViserServer,
    frame_groups: dict[str, list[FrameHandle]],
    state: dict[str, float],
):
    """Add GUI sliders for frame scale control.

    Args:
        server: Viser server
        frame_groups: Dict mapping labels to lists of frame handles
        state: Dict mapping frame group labels to scale values
    """
    sliders = {}

    with server.gui.add_folder("Frame Scale", expand_by_default=False):
        for label, frames in frame_groups.items():
            if not frames:
                continue

            base_sizes = [(frame, frame.axes_length, frame.axes_radius, frame.origin_radius) for frame in frames]

            slider = server.gui.add_slider(
                label,
                min=_FRAME_SCALE_MIN,
                max=_FRAME_SCALE_MAX,
                step=_FRAME_SCALE_STEP,
                initial_value=state.get(label, 1.0),
            )
            sliders[label] = slider

            @slider.on_update
            def _(event, sizes=base_sizes, group_label=label):
                scale = event.target.value
                state[group_label] = scale
                for frame, axes_length, axes_radius, origin_radius in sizes:
                    frame.axes_length = axes_length * scale
                    frame.axes_radius = axes_radius * scale
                    frame.origin_radius = origin_radius * scale

            slider.value = slider.value

        reset_button = server.gui.add_button("Reset Frame Scales")

        @reset_button.on_click
        def _(_):
            for slider in sliders.values():
                slider.value = 1.0
            state.clear()
