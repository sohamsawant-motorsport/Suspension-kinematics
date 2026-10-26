# -*- coding: utf-8 -*-
"""
Side-view silhouette of the car, for drawing behind the anti-geometry
construction.
======================================================================

WHY IT EXISTS
-------------
The anti-dive construction is three straight lines and two dots. On empty axes
that is abstract: you can read the numbers but you cannot see WHERE the force
line passes through the car. Ghosting the real body behind it turns the plot
into something you can point at -- the same thing the reference chassis
drawings do.

WHERE IT COMES FROM
-------------------
`CAD/lamborghini_temerario_gt3_2026.stl` -- the BINARY export, 1 163 029
triangles. Binary matters: it is read with one `np.fromfile` and it gives whole
TRIANGLES. The 356 MB ASCII copy of the same model gives only loose vertices,
and a picture built from corners alone comes out as a cloud rather than a
wireframe.

    mesh frame:  (lateral, VERTICAL, LONGITUDINAL), ground at 0,
                 +longitudinal = FORWARD (confirmed from the profile: the wing
                 and diffuser sit at one end, the splitter tapers at the other)

Neither file is opened at run time. Two layers are baked once:

  car_outline.npz  the SILHOUETTE -- min and max height in each of 420
                   longitudinal bins, twice median-filtered to drop single-bin
                   mesh spikes without changing the shape.
  car_render.npz   the WIREFRAME -- every triangle edge sampled at seven points
                   (24 M samples) and binned in side view, so wheel spokes,
                   brake ducts, panel lines and the wing draw themselves.

THE AXLES COME FROM THE CAD ITSELF
----------------------------------
Not typed in: the contact patches are found as the two places the mesh TOUCHES
THE GROUND, which lands them at -12.935 and +14.030 in mesh units -- a 26.965
wheelbase. That is the calibration both layers are normalised against, so they
register on each other exactly.

WHY IT IS STORED NORMALISED BY WHEELBASE
----------------------------------------
The mesh's own wheelbase is 2696 mm; the hard points say 2645. That ~2 % is the
known mesh accuracy limit (see LIMITATIONS L12), and it matters here: if the
picture were placed at its own scale its wheels would not sit under the plotted
contact patches, and it would quietly contradict the numbers drawn on top of it.

So both layers are stored with the axles at u = +/-0.5 and heights divided by
the mesh wheelbase, and `outline()` scales them onto whatever the LIVE hard
points currently say. The wheels always land on the patches, whatever the
geometry is edited to.

The mesh is an artistic model, not the real car -- this is a backdrop for
reading the construction against, never a source of dimensions.
"""

import os

import numpy as np

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "car_outline.npz")
_CACHE = None


def _load():
    """The normalised envelope, or None if the file is not there."""
    global _CACHE
    if _CACHE is None:
        if not os.path.exists(_PATH):
            _CACHE = False
        else:
            d = np.load(_PATH)
            _CACHE = (np.asarray(d["u"], float), np.asarray(d["lo"], float),
                      np.asarray(d["hi"], float))
    return _CACHE or None


def outline(y_front_mm, y_rear_mm):
    """Silhouette placed on the live axle stations.

    Returns (y, z_lower, z_upper) in the project frame (Y longitudinal, forward
    positive; Z up; mm, ground at 0), or None if the outline file is missing --
    callers should treat that as "just don't draw it".
    """
    d = _load()
    if d is None:
        return None
    u, lo, hi = d
    wb = float(y_front_mm) - float(y_rear_mm)
    mid = 0.5 * (float(y_front_mm) + float(y_rear_mm))
    return mid + u * wb, lo * wb, hi * wb


_RPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "car_render.npz")
_RCACHE = None


def _render():
    """The side-view vertex-density raster, or None.

    An outline alone draws a featureless blob -- no wheels, no arches, nothing
    that reads as a car. This is a real WIREFRAME: every triangle edge in the
    mesh is sampled at seven points and the samples binned in side view, so
    wheel spokes, brake ducts, panel lines and the wing all draw themselves.
    24 million samples off 1.16 M triangles, taken from the BINARY STL (fast to
    read and it gives triangles, not loose vertices -- the ASCII copy gives only
    corners, which is why the first attempt looked like a cloud).

    Same normalisation as the envelope, so the two register exactly.
    """
    global _RCACHE
    if _RCACHE is None:
        if not os.path.exists(_RPATH):
            _RCACHE = False
        else:
            d = np.load(_RPATH)
            _RCACHE = (np.asarray(d["img"], float), float(d["u0"]),
                       float(d["u1"]), float(d["z0"]), float(d["z1"]),
                       float(d["vmax"]))
    return _RCACHE or None


def draw(ax, y_front_mm, y_rear_mm, zorder=0, alpha=0.07, detail=0.42):
    """Ghost the body onto an axis. Returns the drawn extent, or None.

    Two layers: a pale fill for the body mass, then the density raster on top
    for the wheels and structure. Both deliberately faint and under everything
    else -- this is a spatial reference, and it must never compete with the
    construction lines that carry the actual answer.
    """
    got = outline(y_front_mm, y_rear_mm)
    if got is None:
        return None
    y, lo, hi = got
    ax.fill_between(y, lo, hi, color="#9aa2a9", alpha=alpha, lw=0, zorder=zorder)
    ax.plot(y, hi, lw=1.0, color="#5f666c", alpha=0.5, zorder=zorder)
    ax.plot(y, lo, lw=1.0, color="#5f666c", alpha=0.5, zorder=zorder)
    r = _render()
    if r is not None:
        img, u0, u1, z0, z1, vmax = r
        wb = float(y_front_mm) - float(y_rear_mm)
        mid = 0.5 * (float(y_front_mm) + float(y_rear_mm))
        ax.imshow(img, origin="lower", cmap="bone_r", vmin=0.0, vmax=vmax,
                  alpha=detail, interpolation="bilinear", zorder=zorder,
                  extent=[mid + u0 * wb, mid + u1 * wb, z0 * wb, z1 * wb],
                  aspect="auto")
    return float(y.min()), float(y.max()), float(hi.max())


if __name__ == "__main__":
    import config as cfg
    import rear_geometry as rgeo

    yf = float(cfg._load_front_points()["Wheel centre"][1])
    yr = float(rgeo.load_rear()["Wheel centre"][1])
    y, lo, hi = outline(yf, yr)
    print(f"axles at Y {yf:+.1f} / {yr:+.1f} mm  ->  wheelbase {yf - yr:.1f}")
    print(f"outline spans Y {y.min():+.0f} .. {y.max():+.0f} mm "
          f"(overhangs {y.max() - yf:.0f} front, {yr - y.min():.0f} rear)")
    print(f"body height {hi.max():.0f} mm,  {len(y)} bins")
