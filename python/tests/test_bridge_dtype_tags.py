"""C5 step 6: nslpy decodes a result desc by its dtype tag.

`read_output_desc` used to accept only tags 0 and 1, and the gradient
reader `read_f32_output_desc` ignored the tag and read f32 whatever it
said. These build descs over Python-owned buffers, so no model library is
needed.
"""

import ctypes
import struct

import pytest

from nslpy._bridge import NslTensorDesc, read_f32_output_desc, read_output_desc


def make_desc(raw: bytes, shape: list[int], tag: int, keep: list):
    buf = ctypes.create_string_buffer(raw, len(raw))
    dims = (ctypes.c_int64 * max(len(shape), 1))(*shape)
    keep.extend((buf, dims))
    return NslTensorDesc(
        data=ctypes.cast(buf, ctypes.c_void_p),
        shape=dims,
        strides=None,
        ndim=len(shape),
        dtype=tag,
        device_type=0,
        device_id=0,
    )


@pytest.mark.parametrize(
    "tag, raw, want",
    [
        (0, struct.pack("<3d", 1.5, -2.25, 3.0), [1.5, -2.25, 3.0]),
        (1, struct.pack("<3f", 1.5, -2.25, 3.0), [1.5, -2.25, 3.0]),
        (2, struct.pack("<3e", 1.5, -2.25, 3.0), [1.5, -2.25, 3.0]),
        # bf16 = the high 16 bits of the f32.
        (
            3,
            b"".join(struct.pack("<f", v)[2:] for v in (1.5, -2.25, 3.0)),
            [1.5, -2.25, 3.0],
        ),
        (4, struct.pack("<3b", 1, -2, 127), [1, -2, 127]),
        (9, struct.pack("<3i", 1, -2, 2**31 - 1), [1, -2, 2**31 - 1]),
    ],
)
def test_every_c_api_tag_is_decoded_as_itself(tag, raw, want):
    keep: list = []
    assert read_output_desc(make_desc(raw, [3], tag, keep)) == want


def test_an_unknown_tag_is_refused_by_name():
    keep: list = []
    with pytest.raises(ValueError, match="tag 7"):
        read_output_desc(make_desc(b"\x00" * 6, [3], 7, keep))


def test_a_rank0_result_is_a_scalar():
    keep: list = []
    assert read_output_desc(make_desc(struct.pack("<d", 7.5), [], 0, keep)) == 7.5


def test_the_gradient_reader_follows_the_tag():
    # An f64 gradient used to be read as f32 halves.
    keep: list = []
    got = read_f32_output_desc(make_desc(struct.pack("<2d", 0.1, -0.2), [2], 0, keep))
    assert got == [0.1, -0.2]
