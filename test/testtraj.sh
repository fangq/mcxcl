#!/bin/sh
#
# testtraj.sh - step-by-step photon trajectory tests for MCX-CL
#
# usage: ./testtraj.sh [extra mcxcl parameters, such as -G 1 to select a device]
#
# each case under traj_cases/ is simulated with a fixed seed and one photon
# per GPU thread (-A 0 -t <photons> -T 1), so that every photon draws from its
# own reproducible random number stream and the saved trajectory (-D M) is
# deterministic; it is then checked against physical invariants and compared
# to the solver-neutral golden file golden/<case>.json. The golden data are
# copied from MCX (mcx/test/golden): MCX-CL derives the per-thread seeds the
# same way, so both solvers trace the same paths up to single-precision
# round-off, which is tolerated per photon by REFTOL and MAXFORK (see
# trajtest.py: photons with the reference step count must match every step,
# at most MAXFORK of the photons may fork at a reflect/escape decision).
#
# environment variables:
#    MCX:    path to the mcxcl binary (default ../bin/mcxcl)
#    PYTHON: python interpreter with numpy (default python3)
#    REFTOL: position (mm)/relative weight tolerance (default 5e-2)
#    MAXFORK: maximum fraction of photons forking from the golden data (default 0.1)

MCX=${MCX:-$(dirname "$0")/../bin/mcxcl}
PYTHON=${PYTHON:-python3}
REFTOL=${REFTOL:-5e-2}
MAXFORK=${MAXFORK:-0.1}
PARAM="$@"
TESTDIR=$(cd "$(dirname "$0")" && pwd)

case "$MCX" in
    */*) MCX=$(cd "$(dirname "$MCX")" && pwd)/$(basename "$MCX") ;;
esac
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

if ! "$MCX" -L 2>&1 | grep -q 'Device [0-9]'; then
    echo "no OpenCL device is found, skip trajectory tests"
    exit 0
fi

fail=0

for casefile in "$TESTDIR"/traj_cases/*.json; do
    case=$(basename "$casefile" .json)
    echo "test trajectory of case '$case' ... "

    nphoton=$("$PYTHON" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['Session']['Photons']))" "$casefile")
    (cd "$WORKDIR" && "$MCX" -f "$casefile" -A 0 -t "$nphoton" -T 1 -D M -F jnii -s "$case" $PARAM > "$case.log" 2>&1)

    traj="$WORKDIR/${case}_traj.jdt"
    [ -f "$traj" ] || traj="$WORKDIR/${case}_traj.jdat"

    if [ ! -f "$traj" ]; then
        echo "fail to save trajectory, see log below"
        tail -5 "$WORKDIR/$case.log"
        fail=$((fail + 1))
        continue
    fi

    "$PYTHON" "$TESTDIR/trajtest.py" "$traj" --case "$casefile" --ref "$TESTDIR/golden/$case.json" \
        --tol "$REFTOL" --wtol "$REFTOL" --max-fork "$MAXFORK" || fail=$((fail + 1))
done

if [ "$fail" -gt 0 ]; then
    echo "failed $fail trajectory tests"
    exit $fail
fi

echo "passed all trajectory tests!"
