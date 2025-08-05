# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

__all__ = ["ImageAssetPointerChunkParam"]


class ImageAssetPointerChunkParam(TypedDict, total=False):
    format: Required[Literal["png", "jpeg"]]
    """Image format"""

    height: Required[int]
    """Image height in pixels"""

    location: Required[str]
    """Path or URL to the image asset"""

    tokens: Required[int]
    """Number of tokens this image represents"""

    type: Required[Literal["image_asset_pointer"]]

    width: Required[int]
    """Image width in pixels"""
