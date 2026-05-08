# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Opt-in tracer for the layered ``[TS]`` walkthrough in ``LEARN.md``.

Set the ``TS_TRACE`` environment variable to a truthy value
(``1``, ``true``, ``yes``, ``on`` — case-insensitive) to enable every
print statement in the engine / runtime / model / kernel layers of the
print-trace plan.

Default is **off** so production runs and tests pay zero cost: ``ts_log``
becomes a single boolean test on a module-level flag and a no-op return.

Usage::

    from tokenspeed.runtime.utils.ts_trace import ts_log

    ts_log(f"[TS][http] POST /v1/chat/completions model={model}")

The ``[TS][<layer>][<sub>]`` prefix convention is documented in
``LEARN.md`` section 4.
"""

from __future__ import annotations

import os

_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "y", "t"})


def _is_truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in _TRUE_VALUES


TS_TRACE: bool = _is_truthy(os.environ.get("TS_TRACE"))


def ts_log(msg: str) -> None:
    """Print ``msg`` (with ``flush=True``) iff ``TS_TRACE`` is enabled.

    The flush is intentional — these prints are interleaved with output
    from the scheduler subprocess, and Python buffers stdout when piped
    to a file/pager, which would otherwise scramble the cause-and-effect
    timeline the doc is trying to teach.
    """
    if TS_TRACE:
        print(msg, flush=True)
