# -*- coding: utf-8 -*-
"""
RUN EVERY VALIDATION SUITE  -  one gate for the whole geometry stack.
=====================================================================

The project's rule is that no result is trusted until a suite asserts it against
the validated cores and the real CAD. This runner executes all of them and fails
loudly if any single check fails -- run it before extending anything so a
regression in a shared core is caught immediately. Now covers the front axle plus
the rear static geometry.

    python validate_all.py

Each suite lives next to the code it guards:
    scrub_trail   patch / scrub / trail / jacking
    wishbone      full double-wishbone position solve
    geom_edit     KPI/caster -> hard-point inverse
    axle          two-corner roll centre / FVSA / roll migration  (suspension_tool)
    explorer      the GUI's compute layer -- adapter over all the above
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
ST = os.path.join(PROJ, "suspension_tool")

SUITES = [
    ("scrub_trail", HERE, "validate_scrub_trail.py"),
    ("wishbone", HERE, "validate_wishbone.py"),
    ("geom_edit", HERE, "validate_geom_edit.py"),
    ("axle", ST, "validate_axle.py"),
    ("explorer", HERE, "validate_explorer.py"),
    ("rear_geometry", HERE, "validate_rear_geometry.py"),
    ("rear_bump", HERE, "validate_rear_bump.py"),
    ("rear_roll", HERE, "validate_rear_roll.py"),
    ("load_transfer", HERE, "validate_load_transfer.py"),
    ("conventions", HERE, "validate_conventions.py"),
    ("ackermann", HERE, "validate_ackermann.py"),
    ("springs", HERE, "validate_springs.py"),
    ("arb", HERE, "validate_arb.py"),
    ("roll_axis", HERE, "validate_roll_axis.py"),
    ("anti_geometry", HERE, "validate_anti_geometry.py"),
    ("mech_balance", HERE, "validate_mech_balance.py"),
    ("hardpoint_audit", HERE, "validate_hardpoint_audit.py"),
]


def main():
    print("=" * 78)
    print("GEOMETRY VALIDATION - running every suite (front axle + rear static)")
    print("=" * 78)
    total_pass, total_fail, broken = 0, 0, []
    for name, cwd, script in SUITES:
        # capture bytes and decode defensively: the suites print em-dashes and on
        # Windows the child's stdout is cp1252, which is not valid UTF-8
        r = subprocess.run([sys.executable, script], cwd=cwd, capture_output=True)
        out = (r.stdout or b"").decode("utf-8", "replace")
        err = (r.stderr or b"").decode("utf-8", "replace")
        passes = len(re.findall(r"\[PASS\]", out))
        fails = len(re.findall(r"\*\*FAIL\*\*", out))
        total_pass += passes
        total_fail += fails
        ok = r.returncode == 0 and fails == 0
        if not ok:
            broken.append(name)
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:<12s}  {passes:3d} pass  "
              f"{fails:d} fail")
        if not ok:                       # surface the failing lines
            for line in out.splitlines():
                if "**FAIL**" in line or "FAILED" in line:
                    print("          " + line.strip())
            if err.strip():
                print("          " + err.strip().splitlines()[-1])

    print("-" * 78)
    print(f"  TOTAL  {total_pass} pass  {total_fail} fail  "
          f"across {len(SUITES)} suites")
    if broken:
        print("  BROKEN:", ", ".join(broken))
        print("=" * 78)
        raise SystemExit(1)
    print("  ALL SUITES GREEN - geometry is sound; safe to extend.")
    print("=" * 78)


if __name__ == "__main__":
    main()
