"""Step 8.  Build every table in the paper directly from the data files.

No number in the manuscript is typed by hand.  Each table body is written to
`paper/tables/` and pulled in with \\input, so a table can never disagree with
the run that produced it.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
T = os.path.join(ROOT, "paper", "tables")


def load(name):
    with open(os.path.join(D, name)) as f:
        return json.load(f)


def write(name, lines):
    os.makedirs(T, exist_ok=True)
    path = os.path.join(T, name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(path, ROOT))


def sci(x, digits=1):
    """LaTeX scientific notation."""
    x = float(x)
    if x == 0:
        return "0"
    e = 0
    while abs(x) >= 10:
        x /= 10.0
        e += 1
    while abs(x) < 1:
        x *= 10.0
        e -= 1
    m = f"{x:.{digits}f}"
    return f"${m}\\times10^{{{e}}}$"


CLOSED = {
    "lorentzian": ("$1$", 1.0),
    "gaussian": ("$\\pi/2=1.5707963$", 1.5707963267948966),
    "student_t3": ("$4/3=1.3333333$", 4.0 / 3.0),
    "box": ("$\\pi^{2}/4=2.4674011$", 2.4674011002723395),
}
NICE = {"lorentzian": "Lorentzian", "gaussian": "Gaussian",
        "student_t3": "Student $t$, $\\nu=3$", "box": "box",
        "uniform_box": "box", "student_t_nu3": "Student $t$, $\\nu=3$",
        "kuramoto": "overdamped", "conservative": "conservative"}


def table_amplitude():
    d = load("01_parametric.json")["conservative"]
    key = {"lorentzian": "lorentzian", "gaussian": "gaussian",
           "student_t_nu3": "student_t3", "uniform_box": "box"}
    rows = []
    for name in ("lorentzian", "gaussian", "student_t_nu3", "uniform_box"):
        meas = float(d[name]["A_num"])
        closed, exact = CLOSED[key[name]]
        rows.append(f"{NICE[name]} & {closed} & ${meas:.7f}$ & "
                    f"{sci(abs(meas - exact) / exact)}\\\\")
    write("amplitude.tex", rows)


def table_beta():
    d = load("02_universality_line.json")
    rows = []
    for r in d["table"]:
        if r["s"] in (1.4, 2.8):
            continue
        s = r["s"]
        pred = r["predicted"]
        frac = {1.5: "$2$", 1.8: "$5/4$", 2.0: "$1$", 2.2: "$5/6$",
                2.5: "$2/3$", 3.0: "$1/2$", 3.5: "$1/2$", 4.0: "$1/2$",
                6.0: "$1/2$"}.get(s, f"${pred:.6f}$")
        rows.append(f"${s}$ & ${float(r['beta']):.8g}$ & {frac} & "
                    f"{sci(r['rel_dev'])}\\\\")
    rows.append("\\hline")
    for key, label in (("kuramoto", "overdamped"),
                       ("gaussian", "Gaussian kernel")):
        b = float(d["reference"][key]["beta"])
        rows.append(f"{label} & ${b:.8g}$ & $1/2$ & {sci(abs(b - 0.5) / 0.5)}\\\\")
    write("beta.tex", rows)


def table_general_amplitude():
    d = load("02_universality_line.json").get("amplitude", [])
    rows = []
    for r in d:
        rows.append(f"{NICE.get(r['line'], r['line'])} & ${r['s']}$ & "
                    f"${float(r['A_pred']):.9g}$ & ${float(r['A_meas']):.9g}$ & "
                    f"{sci(r['rel_dev'])}\\\\")
    write("gen_amplitude.tex", rows)


def table_heterogeneous():
    d = load("06_coupling_disorder.json")["rows"]
    rows = []
    for r in d:
        pub = ("$%.6g$" % r["beta_published"]) if r["beta_published"] is not None \
            else "---"
        rows.append(f"{NICE[r['base']]} & ${r['gamma']}$ & ${r['eta']}$ & "
                    f"${r['s_predicted']:.4g}$ & ${r['s_measured']:.5g}$ & "
                    f"${r['beta_measured']:.8g}$ & {pub}\\\\")
    write("heterogeneous.tex", rows)


def table_vlasov():
    d = load("04_meanfield_check.json")
    rows = []
    labels = {"grid": "grid", "time": "run length", "cutoff": "cutoff"}
    for key in ("grid", "time", "cutoff"):
        for r in sorted(d[key], key=lambda q: (q["r"], q["M_core"], q["T"],
                                               q["freefac"])):
            rows.append(
                f"{labels[key]} & ${r['r']}$ & ${r['M_core']}$ & "
                f"${r['T']:.0f}$ & ${r['freefac']:.0f}$ & "
                f"{sci(r['rel_dev'])} & {sci(r['drift'])} & "
                f"{sci(r['osc_sd'] / r['R_law'])}\\\\")
    write("vlasov.tex", rows)


def main():
    table_amplitude()
    table_beta()
    table_general_amplitude()
    table_heterogeneous()
    table_vlasov()


if __name__ == "__main__":
    main()
