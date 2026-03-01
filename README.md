<p align="center">
  <img src="https://raw.githubusercontent.com/henrynoyes/robot-see/refs/heads/master/docs/robot-see-logo.svg" width="60%"/>
</p>

<h3 align="center">
  A visualization tool for robot models
</h3>

[robot-see-demo.webm](https://github.com/user-attachments/assets/18fb12b7-5c99-460d-8660-9090263749c3)

## Installation

To install globally as a [uv tool](https://docs.astral.sh/uv/guides/tools/),
```sh
uv tool install git+https://github.com/henrynoyes/robot-see.git
```

## Usage

![help](https://raw.githubusercontent.com/henrynoyes/robot-see/refs/heads/master/docs/help.png)

`robot-see` is a single command that can be called on a path to any robot model file. It parses the robot model into a unified internal representation and constructs a local web visualization with GUI controls. The GUI allows the user to toggle the visibility of different geometry groups, actuate individual joints, and scale the frame groups for detailed visual control. The tool is designed to make the creation of robot models easier and more intuitive, with less mistakes flying under the radar.

### IDE Integration

Since `robot-see` serves the visualization on localhost, any IDE with an integrated browser window can load the scene side-by-side. For example, VSCode supports this through the `Simple Browser` command: `CTRL+SHIFT+P` -> `Simple Browser: Show` -> http://localhost:8080 (or your specified port). To enable live reloading while editing a model, pass in the `--watch` flag.

![ide integration](https://raw.githubusercontent.com/henrynoyes/robot-see/refs/heads/master/docs/ide-integration.png)

## Implementation

At its core, `robot-see` performs two actions: **parsing** and **scene construction**. The first stage is handled by the classes in `parsers.py`, which extract the relevant information from a given robot model file. This information is stored in a `Robot` object, which serves as the unified internal representation for robot models. The heavy lifting is performed by `lxml` for the XML formats (URDF, SDF, MJCF), and `usd-core` for the Isaac USD models. For more information on the conventions, [see here](https://github.com/henrynoyes/robot-diff?tab=readme-ov-file#conventions).

The visualization is constructed using [`viser`](https://github.com/nerfstudio-project/viser), a 3D visualization library tailored to robotics applications. The kinematic tree from the `Robot` is converted into a scene graph of frames with associated geometries. Meshes are handled by [`trimesh`](https://github.com/mikedh/trimesh) to support features such as uv texturing.

## Acknowledgements

A huge thanks to the developers of [`viser`](https://github.com/nerfstudio-project/viser), which this project leverages for visualization. The parsers and internal representation were adapted from [`robot-diff`](https://github.com/henrynoyes/robot-diff).

## Contributing

Contributions are welcome! I have done my best to benchmark against models from a variety of sources, but there are sure to be edge cases that were left unconsidered. In particular, the SDF and Isaac USD parsers could probably be more robust.

## Development

To manually lint/format,
```sh
uv run ruff check --fix .
uv run ruff format .
```
Ruff is also configured as a pre-commit hook

To type check,
```sh
uv run ty check
```

To build a release,
```sh
uv build # generates wheel and source in dist/
```
