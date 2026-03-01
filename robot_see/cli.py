import time
import warnings
from collections.abc import Callable
from pathlib import Path

import tyro
import viser
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .gui import GUIState, add_frame_scale_sliders, add_joint_sliders, add_visibility_folder
from .parsers import IsaacUSDParser, MJCFParser, Parser, SDFParser, URDFParser
from .scene import RobotScene

PARSER_MAP = {
    ".urdf": URDFParser,
    ".sdf": SDFParser,
    ".xml": MJCFParser,
    ".usd": IsaacUSDParser,
    ".usda": IsaacUSDParser,
    ".usdc": IsaacUSDParser,
}


def _get_parser(path: Path) -> Parser:
    """Get parser based on file extension.

    Args:
        path: Path to robot model file

    Returns:
        Parser instance for the given file type

    Raises:
        ValueError: If file extension is not supported
    """
    parser_cls = PARSER_MAP.get(path.suffix.lower())

    if parser_cls is None:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    return parser_cls(path)


class _ReloadHandler(FileSystemEventHandler):
    """Handler for file modification events with debouncing"""

    def __init__(self, path: Path, callback: Callable[[], None]):
        self.path = path
        self.callback = callback
        self.last_mtime = path.stat().st_mtime

    def on_modified(self, event):
        """Trigger reload when file is modified"""
        if Path(event.src_path).resolve() == self.path.resolve():
            try:
                current_mtime = self.path.stat().st_mtime
            except FileNotFoundError:
                return
            if current_mtime != self.last_mtime:
                self.last_mtime = current_mtime
                print(f"Change detected in {self.path.name}, reloading...")
                self.callback()


def _create_file_watcher(path: Path, callback: Callable[[], None]) -> BaseObserver:
    """Create and start a file watcher

    Args:
        path: File to watch for changes
        callback: Function to call when file is modified

    Returns:
        Started Observer instance
    """
    observer = Observer()
    handler = _ReloadHandler(path, callback)
    observer.schedule(handler, str(path.parent), recursive=False)
    observer.start()
    return observer


def _load_model(server: viser.ViserServer, path: Path, state: GUIState) -> None:
    """Load robot model and build GUI

    Args:
        server: Viser server
        path: Path to robot model file
        state: View state to preserve across reloads
    """
    robot = _get_parser(path).parse()

    server.scene.reset()
    server.gui.reset()

    server.initial_camera.position = (1.0, 1.0, 1.0)

    robot_scene = RobotScene(server, robot)

    reload_button = server.gui.add_button("Reload Model")
    reload_button.on_click(lambda _: _load_model(server, path, state))

    reset_button = server.gui.add_button("Reset Model")

    @reset_button.on_click
    def _(_):
        _load_model(server, path, GUIState())

    with server.atomic():
        _build_gui(robot_scene, state)
        server.flush()

    print(f"Loaded: {robot.name}")


def _build_gui(robot_scene: RobotScene, state: GUIState) -> None:
    """Populate scene panel with GUI elements

    Args:
        robot_scene: RobotScene containing scene graph handles
        state: View state to synchronize
    """
    server = robot_scene.server
    with server.gui.add_folder("Visibility"):
        link_frames_checkbox = server.gui.add_checkbox("Link Frames", initial_value=state.link_frames_visible)

        @link_frames_checkbox.on_update
        def _(event):
            state.link_frames_visible = event.target.value
            for frame in robot_scene.link_frames.values():
                frame.show_axes = event.target.value

        link_frames_checkbox.value = link_frames_checkbox.value

        add_visibility_folder(server, "Visual", robot_scene.visual_frames, state.visual)
        add_visibility_folder(server, "Collision", robot_scene.collision_frames, state.collision)
        add_visibility_folder(server, "Inertial", robot_scene.inertial_frames, state.inertial)

    add_joint_sliders(server, robot_scene.robot.joints, robot_scene.update_joint, state.joint_positions)

    frame_groups = {
        "Link Frames": list(robot_scene.link_frames.values()),
        "Visual Frames": robot_scene.visual_frames["origin"],
        "Collision Frames": robot_scene.collision_frames["origin"],
        "Inertial Frames": robot_scene.inertial_frames["origin"] + robot_scene.inertial_frames["principal"],
    }
    add_frame_scale_sliders(server, frame_groups, state.frame_scales)


def main(
    path: Path,
    /,
    port: int = 8080,
    watch: tyro.conf.FlagCreatePairsOff[bool] = False,
    quiet: tyro.conf.FlagCreatePairsOff[bool] = False,
) -> None:
    """Visualize a robot model. Supports URDF (.urdf), SDF (.sdf), MJCF (.xml), and USD (.usd) files.

    Args:
        path: Path to robot model file
        port: Port for visualization server
        watch: Automatically reload model on file changes
        quiet: Suppress warnings
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if quiet:
        warnings.filterwarnings("ignore")

    state = GUIState()
    server = viser.ViserServer(port=port)
    _load_model(server, path, state)

    if watch:
        observer = _create_file_watcher(path, lambda: _load_model(server, path, state))
        print(f"Watching {path.name} for changes...")

    try:
        while True:
            time.sleep(1.0)
    finally:
        if watch:
            observer.stop()
            observer.join()


def tyro_cli():
    tyro.cli(main, prog="robot-see")


if __name__ == "__main__":
    tyro_cli()
