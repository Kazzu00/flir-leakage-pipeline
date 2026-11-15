"""Parse explicit name fields without assigning videos, timestamps or sequences."""

import re
from pathlib import PurePosixPath


def parse_name_fields(name: str) -> dict:
    """A prefix's digits are never mistaken for a frame/index or scene token."""
    stem = PurePosixPath(name.replace("\\", "/")).stem
    scene = re.fullmatch(
        r"(?:(?P<prefix>v\d+)_)?frame_(?P<index>\d+)_scene(?P<scene>\d+)", stem
    )
    if scene:
        return dict(
            family="scene_named",
            prefix=scene["prefix"],
            nominal_index=int(scene["index"]),
            scene_token=int(scene["scene"]),
            temporal_identity_verified=False,
        )
    video = re.fullmatch(r"(?P<family>video_\d+min)_(?P<index>\d+)", stem)
    if video:
        return dict(
            family=video["family"],
            prefix=None,
            nominal_index=int(video["index"]),
            scene_token=None,
            temporal_identity_verified=False,
        )
    frame = re.fullmatch(r"frame_(?P<index>\d+)", stem)
    return dict(
        family="frame_named" if frame else "unresolved",
        prefix=None,
        nominal_index=int(frame["index"]) if frame else None,
        scene_token=None,
        temporal_identity_verified=False,
    )
