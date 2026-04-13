"""
P452 Project 1 — Universal Quantum Simulator
Streamlit web app backed by Qiskit-Aer (10-qubit AerSimulator).

Modes
-----
1. Custom Circuit  — build any circuit interactively (up to 10 qubits)
2. Teleportation   — 3-qubit quantum-teleportation protocol (Q2.1)
3. Fermi-Hubbard   — 4-qubit Trotter simulation via Jordan-Wigner (Phase 3)
"""

import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator

# ============================================================
#  BACKEND HELPERS
# ============================================================

def _simulator(method="automatic"):
    return AerSimulator(method=method)


def run_shots(qc: QuantumCircuit, shots: int = 2048) -> dict:
    """Attach measure_all and return counts dict."""
    qc_m = qc.copy()
    qc_m.measure_all()
    result = _simulator().run(qc_m, shots=shots).result()
    return result.get_counts()


def run_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Return statevector as numpy array (no measurements allowed)."""
    qc_sv = qc.copy()
    qc_sv.save_statevector()
    result = _simulator("statevector").run(qc_sv).result()
    return np.array(result.get_statevector())


def draw_circuit(qc: QuantumCircuit, scale: float = 0.4, fold: int = -1) -> plt.Figure:
    """Render circuit with Qiskit's Matplotlib drawer."""
    fig = qc.draw(output="mpl", fold=fold, style="clifford", scale=scale)
    return fig


def circuit_image_bytes(qc: QuantumCircuit, max_width_in: float = 9.0) -> bytes:
    """
    Draw the circuit, cap its width at max_width_in inches (preserving aspect
    ratio), render to PNG bytes, and close the figure.  Use with st.image().
    """
    fig = qc.draw(output="mpl", fold=-1, style="clifford", scale=0.4)
    w, h = fig.get_size_inches()
    if w > max_width_in:
        fig.set_size_inches(max_width_in, h * max_width_in / w)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def counts_to_probs(counts: dict) -> dict:
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items())}


def plot_histogram(counts: dict, title: str = "Measurement Results") -> plt.Figure:
    probs = counts_to_probs(counts)
    n_states = len(probs)

    if n_states > 64:
        # Too many states to show individually — display top-32 by probability
        # plus an "other" bucket, with a note
        top_items = sorted(probs.items(), key=lambda x: -x[1])[:32]
        other_p   = 1.0 - sum(v for _, v in top_items)
        states = [k for k, _ in top_items]
        vals   = [v for _, v in top_items]
        if other_p > 1e-6:
            states.append(f"… +{n_states-32} more")
            vals.append(other_p)
        note = f"(Showing top 32 of {n_states} states)"
    else:
        states, vals = zip(*probs.items()) if probs else ([], [])
        note = ""

    fig, ax = plt.subplots(figsize=(max(8, len(states) * 0.55 + 2), 4))
    cmap = plt.cm.viridis(np.linspace(0.25, 0.85, len(states)))
    bars = ax.bar(states, vals, color=cmap, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Basis State", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    full_title = f"{title}\n{note}" if note else title
    ax.set_title(full_title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25 if vals else 1.1)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    for bar, v in zip(bars, vals):
        if v > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    return fig


# ============================================================
#  GATE APPLICATION (custom circuit mode)
# ============================================================

_SINGLE = ["X", "Y", "Z", "H", "S", "Sdg", "T", "Tdg", "Rx", "Ry", "Rz"]
_DOUBLE = ["CNOT", "CY", "CZ", "SWAP", "CH"]
_TRIPLE = ["Toffoli", "CSWAP"]
_ALL    = _SINGLE + _DOUBLE + _TRIPLE + ["Barrier"]


def apply_gate(qc: QuantumCircuit, g: dict):
    n, qb, p = g["name"], g["qubits"], g["params"]
    if   n == "X":       qc.x(qb[0])
    elif n == "Y":       qc.y(qb[0])
    elif n == "Z":       qc.z(qb[0])
    elif n == "H":       qc.h(qb[0])
    elif n == "S":       qc.s(qb[0])
    elif n == "Sdg":     qc.sdg(qb[0])
    elif n == "T":       qc.t(qb[0])
    elif n == "Tdg":     qc.tdg(qb[0])
    elif n == "Rx":      qc.rx(p[0], qb[0])
    elif n == "Ry":      qc.ry(p[0], qb[0])
    elif n == "Rz":      qc.rz(p[0], qb[0])
    elif n == "CNOT":    qc.cx(qb[0], qb[1])
    elif n == "CY":      qc.cy(qb[0], qb[1])
    elif n == "CZ":      qc.cz(qb[0], qb[1])
    elif n == "SWAP":    qc.swap(qb[0], qb[1])
    elif n == "CH":      qc.ch(qb[0], qb[1])
    elif n == "Toffoli": qc.ccx(qb[0], qb[1], qb[2])
    elif n == "CSWAP":   qc.cswap(qb[0], qb[1], qb[2])
    elif n == "Barrier": qc.barrier()


def build_custom(n_qubits: int, gates: list) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for g in gates:
        # Skip gates whose qubit indices exceed the current circuit size
        # (can happen if the user reduces n after adding gates)
        if all(q < n_qubits for q in g["qubits"]):
            apply_gate(qc, g)
    return qc


# ============================================================
#  TELEPORTATION  (3-qubit)
# ============================================================

def build_tele_display(theta: float, phi: float) -> QuantumCircuit:
    """Visual circuit (with if_test corrections, no final measure)."""
    qr = QuantumRegister(3, "q")
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr)

    qc.ry(theta, qr[0])
    if phi != 0.0:
        qc.rz(phi, qr[0])
    qc.barrier(label=" State Prep ")

    qc.h(qr[1])
    qc.cx(qr[1], qr[2])
    qc.barrier(label=" Bell Prep ")

    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.barrier(label=" Bell Meas ")

    qc.measure(qr[0], cr[0])
    qc.measure(qr[1], cr[1])

    with qc.if_test((cr[1], 1)):
        qc.x(qr[2])
    with qc.if_test((cr[0], 1)):
        qc.z(qr[2])
    return qc


def run_tele(theta: float, phi: float, shots: int = 1024) -> dict:
    """Run full teleportation, measure all three qubits, return counts."""
    qc = QuantumCircuit(3, 3)

    qc.ry(theta, 0)
    if phi != 0.0:
        qc.rz(phi, 0)
    qc.barrier()

    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()

    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)

    with qc.if_test((qc.cregs[0][1], 1)):
        qc.x(2)
    with qc.if_test((qc.cregs[0][0], 1)):
        qc.z(2)

    qc.measure(2, 2)
    result = _simulator().run(qc, shots=shots).result()
    return result.get_counts()


# ============================================================
#  FERMI-HUBBARD  (4-qubit, 2-site, Jordan-Wigner)
# ============================================================
#
#  Qubit map: q0 = site-1 ↑,  q1 = site-1 ↓,
#             q2 = site-2 ↑,  q3 = site-2 ↓
#
#  After Jordan-Wigner transform the Hamiltonian becomes:
#
#  H = (J/2)[Z₁(X₀X₂ + Y₀Y₂) + Z₂(X₁X₃ + Y₁Y₃)]
#    + (U/4)[(I-Z₀-Z₁+Z₀Z₁) + (I-Z₂-Z₃+Z₂Z₃)]
#
#  First-order Trotter: e^{-iHτ} ≈ e^{-iH_hop↑τ} e^{-iH_hop↓τ} e^{-iH_intτ}
#
#  Each Pauli-string exponential e^{-iθP₀P₁P₂} is implemented with
#  basis-rotation gates + CNOT ladder + Rz(2θ).
# ============================================================

def _pauli_exp_xzx(qc, i, j, k, theta):
    """e^{-iθ X_i Z_j X_k}: H → CNOT ladder → Rz → reverse."""
    qc.h(i); qc.h(k)
    qc.cx(i, j); qc.cx(k, j)
    qc.rz(2 * theta, j)
    qc.cx(k, j); qc.cx(i, j)
    qc.h(i); qc.h(k)


def _pauli_exp_yzy(qc, i, j, k, theta):
    """e^{-iθ Y_i Z_j Y_k}: Sdg H → CNOT ladder → Rz → H S."""
    qc.sdg(i); qc.h(i)
    qc.sdg(k); qc.h(k)
    qc.cx(i, j); qc.cx(k, j)
    qc.rz(2 * theta, j)
    qc.cx(k, j); qc.cx(i, j)
    qc.h(i); qc.s(i)
    qc.h(k); qc.s(k)


def fh_trotter_step(qc: QuantumCircuit, J: float, U: float, dt: float):
    """Append one first-order Lie-Trotter step to qc (4 qubits)."""
    tJ = J * dt / 2   # hopping angle
    tU = U * dt / 4   # interaction angle

    # --- spin-up hopping (q0 ↔ q2, Z-string through q1) ---
    _pauli_exp_xzx(qc, 0, 1, 2, tJ)
    _pauli_exp_yzy(qc, 0, 1, 2, tJ)

    # --- spin-down hopping (q1 ↔ q3, Z-string through q2) ---
    _pauli_exp_xzx(qc, 1, 2, 3, tJ)
    _pauli_exp_yzy(qc, 1, 2, 3, tJ)

    # --- on-site interaction, site 1 (q0, q1) ---
    # e^{-i(U/4)(I-Z₀-Z₁+Z₀Z₁)τ}  (ignore global phase from I)
    qc.rz(-2 * tU, 0)   # e^{+i tU Z₀}
    qc.rz(-2 * tU, 1)   # e^{+i tU Z₁}
    qc.cx(0, 1)
    qc.rz(2 * tU, 1)    # e^{-i tU Z₀Z₁}
    qc.cx(0, 1)

    # --- on-site interaction, site 2 (q2, q3) ---
    qc.rz(-2 * tU, 2)
    qc.rz(-2 * tU, 3)
    qc.cx(2, 3)
    qc.rz(2 * tU, 3)
    qc.cx(2, 3)


def _state_to_idx(s: str) -> int:
    """Convert qubit-string 'q0q1q2q3' to little-endian integer index."""
    return sum(int(s[i]) * (2 ** i) for i in range(len(s)))


def simulate_fh(J: float, U: float, t_max: float, n_steps: int,
                init_state: str, track: list) -> tuple:
    """
    Simulate Fermi-Hubbard dynamics using exact unitary evolution.

    Strategy: compute the 16×16 unitary of ONE Trotter step via
    qiskit.quantum_info.Operator, then apply it n_steps times.
    This is exact (no shot noise) and very fast for 4 qubits.

    Returns
    -------
    times  : np.ndarray  shape (n_steps+1,)
    probs  : dict[state_str -> np.ndarray of probabilities]
    """
    dt = t_max / n_steps

    # Build Trotter-step unitary (16×16 complex matrix)
    qc_step = QuantumCircuit(4)
    fh_trotter_step(qc_step, J, U, dt)
    U_mat = np.array(Operator(qc_step).data)

    # Initial statevector (computational basis)
    sv = np.zeros(16, dtype=complex)
    sv[_state_to_idx(init_state)] = 1.0

    state_idx = {s: _state_to_idx(s) for s in track}
    times = np.zeros(n_steps + 1)
    probs = {s: np.zeros(n_steps + 1) for s in track}

    for s in track:
        probs[s][0] = abs(sv[state_idx[s]]) ** 2

    for step in range(1, n_steps + 1):
        sv = U_mat @ sv
        times[step] = step * dt
        for s in track:
            probs[s][step] = abs(sv[state_idx[s]]) ** 2

    return times, probs


# ============================================================
#  STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(
        page_title="P452 Quantum Simulator",
        page_icon="⚛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("⚛️  P452 — Universal Quantum Simulator")
    st.caption("10-qubit backend powered by **Qiskit-Aer**  ·  P452 Project 1")

    st.sidebar.title("Mode")
    mode = st.sidebar.radio(
        "Select mode",
        ["🔧  Custom Circuit", "📡  Teleportation", "⚛️  Fermi-Hubbard"],
        label_visibility="collapsed",
    )

    if mode == "🔧  Custom Circuit":
        _ui_custom()
    elif mode == "📡  Teleportation":
        _ui_tele()
    else:
        _ui_fh()


# ------------------------------------------------------------------
#  CUSTOM CIRCUIT
# ------------------------------------------------------------------

def _ui_custom():
    st.header("🔧  Custom Circuit Builder")

    if "gates" not in st.session_state:
        st.session_state.gates = []
    if "cust_n" not in st.session_state:
        st.session_state.cust_n = 3

    left, right = st.columns([1, 2])

    with left:
        n = st.slider("Number of qubits", 1, 10,
                      st.session_state.cust_n, key="cust_nq")
        st.session_state.cust_n = n  # persist across tab switches
        st.markdown("---")
        st.subheader("Add a gate")

        gate = st.selectbox("Gate", _ALL, key="cust_gate")

        if gate in _SINGLE:
            q0 = st.selectbox("Qubit", range(n), key="cust_q0")
            qb = [q0]
        elif gate in _DOUBLE:
            q0 = st.selectbox("Control", range(n), key="cust_q0")
            q1 = st.selectbox("Target",
                              [q for q in range(n) if q != q0], key="cust_q1")
            qb = [q0, q1]
        elif gate in _TRIPLE:
            q0 = st.selectbox("Control 1", range(n), key="cust_q0")
            opts1 = [q for q in range(n) if q != q0]
            q1 = st.selectbox("Control 2" if gate == "Toffoli" else "Control",
                              opts1, key="cust_q1")
            opts2 = [q for q in opts1 if q != q1]
            q2 = st.selectbox("Target", opts2, key="cust_q2")
            qb = [q0, q1, q2]
        else:
            qb = []

        params = []
        if gate in ("Rx", "Ry", "Rz"):
            ang = st.slider("Angle θ (rad)", -2 * np.pi, 2 * np.pi,
                            np.pi / 2, 0.01, format="%.3f", key="cust_ang")
            st.latex(rf"\theta = {ang:.4f}")
            params = [ang]

        c1, c2 = st.columns(2)
        with c1:
            if st.button("➕ Add Gate", type="primary", use_container_width=True):
                if all(q < n for q in qb):
                    st.session_state.gates.append(
                        {"name": gate, "qubits": qb, "params": params}
                    )
        with c2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.gates.clear()
                st.rerun()

        if st.session_state.gates:
            st.markdown("---")
            st.subheader("Gate list")
            for i, g in enumerate(st.session_state.gates):
                col_g, col_d = st.columns([4, 1])
                ps = f"({g['params'][0]:.3f})" if g["params"] else ""
                qs = str(g["qubits"]) if g["qubits"] else ""
                col_g.text(f"{i:2d}: {g['name']}{ps}  q{qs}")
                if col_d.button("✕", key=f"del_{i}"):
                    st.session_state.gates.pop(i)
                    st.rerun()

        st.markdown("---")
        shots = st.slider("Shots", 256, 8192, 2048, 256, key="cust_shots")
        run = st.button("▶️  Run Simulation", type="primary",
                        use_container_width=True, key="cust_run")

    with right:
        st.subheader("Circuit Diagram")
        qc = build_custom(n, st.session_state.gates)
        try:
            st.pyplot(draw_circuit(qc))
        except Exception as e:
            st.error(f"Render error: {e}")

        if run:
            st.subheader("Measurement Histogram")
            with st.spinner("Simulating…"):
                try:
                    counts = run_shots(qc, shots)
                    st.pyplot(plot_histogram(counts,
                              f"Measurement results  ({shots} shots)"))
                    probs = counts_to_probs(counts)
                    top = sorted(probs.items(), key=lambda x: -x[1])[:12]
                    st.markdown("**Top states:**")
                    for state, p in top:
                        st.text(f"  |{state}⟩   p = {p:.4f}   "
                                f"({counts[state]} counts)")
                except Exception as e:
                    st.error(f"Simulation error: {e}")


# ------------------------------------------------------------------
#  TELEPORTATION
# ------------------------------------------------------------------

def _ui_tele():
    st.header("📡  Quantum Teleportation  (3-qubit)")

    with st.expander("📖  Protocol", expanded=False):
        st.markdown(r"""
**Quantum teleportation** transfers an unknown qubit state
$|\psi\rangle = \cos(\theta/2)|0\rangle + e^{i\varphi}\sin(\theta/2)|1\rangle$
from **Alice** (q₀) to **Bob** (q₂) using a shared Bell pair and 2 classical bits.

| Stage | Gates | Qubits |
|---|---|---|
| State prep | Ry(θ), Rz(φ) | q₀ |
| Bell pair | H, CNOT | q₁, q₂ |
| Bell measurement | CNOT, H, Measure | q₀, q₁ |
| Feed-forward correction | X (if c₁=1), Z (if c₀=1) | q₂ |
        """)

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Input state")
        preset = st.selectbox("Preset", [
            "Custom",
            "|0⟩  (θ=0)",
            "|1⟩  (θ=π)",
            "|+⟩  (θ=π/2)",
            "Q2.1: (2|0⟩+|1⟩)/√5",
        ])
        if preset == "|0⟩  (θ=0)":
            theta, phi = 0.0, 0.0
        elif preset == "|1⟩  (θ=π)":
            theta, phi = float(np.pi), 0.0
        elif preset == "|+⟩  (θ=π/2)":
            theta, phi = float(np.pi / 2), 0.0
        elif preset == "Q2.1: (2|0⟩+|1⟩)/√5":
            theta = float(2 * np.arctan2(1, 2))
            phi   = 0.0
        else:
            theta = float(st.slider("θ (rad)", 0.0, float(np.pi),
                                    float(np.pi / 2), 0.01, format="%.3f"))
            phi   = float(st.slider("φ (rad)", 0.0, float(2 * np.pi),
                                    0.0, 0.01, format="%.3f"))

        a = np.cos(theta / 2)
        b = np.exp(1j * phi) * np.sin(theta / 2)
        st.latex(
            rf"|\psi\rangle = {a:.3f}|0\rangle "
            rf"+ ({b.real:.3f}{b.imag:+.3f}i)|1\rangle"
        )
        st.markdown(f"Expected **P(Bob=|0⟩)** = {abs(a)**2:.4f}")

        shots = st.slider("Shots", 256, 4096, 1024, key="tele_shots")
        run = st.button("▶️  Run Teleportation", type="primary",
                        use_container_width=True, key="tele_run")

    with right:
        st.subheader("Circuit")
        try:
            st.pyplot(draw_circuit(build_tele_display(theta, phi)))
        except Exception as e:
            st.error(f"Render error: {e}")

        st.markdown("""
**Classical feed-forward corrections** (applied to Bob's qubit q₂):

| Measurement outcome | Correction applied to q₂ |
|---|---|
| q₁ = 1  (c₁ bit) | **X gate** — bit-flip correction |
| q₀ = 1  (c₀ bit) | **Z gate** — phase-flip correction |

The double-line wires in the diagram carry these classical bits from Alice's measurements to Bob's correction gates.
        """)

        if run:
            st.subheader("Results")
            with st.spinner("Running…"):
                try:
                    counts = run_tele(theta, phi, shots)
                    # counts key = "q2 q1 q0" (3-bit, MSB=q2=Bob)
                    bob0 = sum(v for k, v in counts.items() if k[0] == "0")
                    bob1 = sum(v for k, v in counts.items() if k[0] == "1")
                    total = bob0 + bob1 or 1

                    c1, c2 = st.columns(2)
                    c1.metric("P(Bob = |0⟩)", f"{bob0/total:.4f}",
                              delta=f"Expected {abs(a)**2:.4f}")
                    c2.metric("P(Bob = |1⟩)", f"{bob1/total:.4f}",
                              delta=f"Expected {abs(b)**2:.4f}")

                    st.pyplot(plot_histogram(
                        counts, f"Full 3-qubit histogram  ({shots} shots)"))
                    st.caption(
                        "Bit order in labels: **q₂ q₁ q₀** "
                        "(q₂ = Bob's qubit, leftmost)"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")


# ------------------------------------------------------------------
#  FERMI-HUBBARD
# ------------------------------------------------------------------

def _ui_fh():
    st.header("⚛️  Fermi-Hubbard Model  (2-site, 4 qubits)")

    with st.expander("📖  Physics & Mapping", expanded=False):
        st.markdown(r"""
**Qubit layout** (Jordan-Wigner):
$q_0 = \text{site 1 }{\uparrow}$,
$q_1 = \text{site 1 }{\downarrow}$,
$q_2 = \text{site 2 }{\uparrow}$,
$q_3 = \text{site 2 }{\downarrow}$

**Mapped Hamiltonian:**
$$H = \frac{J}{2}\bigl[Z_1(X_0 X_2+Y_0 Y_2)+Z_2(X_1 X_3+Y_1 Y_3)\bigr]
     +\frac{U}{4}\bigl[(I-Z_0-Z_1+Z_0 Z_1)+(I-Z_2-Z_3+Z_2 Z_3)\bigr]$$

**Trotterization** (first order):
$e^{-iH\tau}\approx e^{-iH_{\uparrow}\tau}\,e^{-iH_{\downarrow}\tau}\,e^{-iH_U\tau}$

Each Pauli-string exponential $e^{-i\theta X_i Z_j X_k}$ is implemented with
basis-rotation gates + CNOT ladder + $R_z(2\theta)$.
        """)

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Parameters")

        J      = st.slider("Hopping J", 0.0, 3.0, 1.0, 0.05, key="fh_J")
        U      = st.slider("Interaction U", 0.0, 20.0, 0.0, 0.5, key="fh_U")

        # Suggest a sensible t_max: for |1100⟩ with U>0, effective rate is J²/U
        if U > 0 and J > 0:
            t_suggested = float(np.pi * U / J**2)   # one Mott oscillation
        elif J > 0:
            t_suggested = float(np.pi / (2 * J))    # free-hopping peak
        else:
            t_suggested = float(np.pi)

        t_max  = float(st.slider("Max time τ_max", 0.1, 120.0,
                                 min(t_suggested, 120.0),
                                 0.5, format="%.1f", key="fh_t"))
        n_steps = st.slider("Trotter steps", 20, 600, 150, 10, key="fh_n")

        init = st.selectbox("Initial state", ["1000", "1100", "0101", "1010"],
                            key="fh_init")
        info = {
            "1000": "Site 1↑ occupied  → Q3.2 (set U=0)",
            "1100": "Site 1↑↓ occupied → Q3.3 (set U=10)",
            "0101": "Site 1↓ + Site 2↓",
            "1010": "Site 1↑ + Site 2↑",
        }
        st.info(info.get(init, ""))

        # Dynamic timescale guidance
        if init == "1100" and J > 0:
            t_rabi  = float(np.pi / (2 * J))          # free-hopping peak
            t_mott  = float(np.pi * U / (2 * J**2)) if U > 0 else t_rabi
            if U / J > 2:
                st.warning(
                    f"⚠️ **Mott regime** (U/J = {U/J:.1f}):  \n"
                    f"Doublon tunneling is suppressed and slow.  \n"
                    f"Set **τ_max ≥ {t_mott:.1f}** to see the oscillation  \n"
                    f"(effective rate ~ J²/U = {J**2/U:.3f})."
                )

        if init == "1000":
            track  = ["1000", "0010"]
            labels = {"1000": "Site 1↑  |1000⟩", "0010": "Site 2↑  |0010⟩"}
        elif init == "1100":
            track  = ["1100", "0011"]
            labels = {"1100": "Doublon Site 1  |1100⟩",
                      "0011": "Doublon Site 2  |0011⟩"}
        else:
            track  = [init]
            labels = {init: f"|{init}⟩"}

        run_sim  = st.button("▶️  Run Simulation", type="primary",
                             use_container_width=True, key="fh_run")
        show_ckt = st.button("🔬  Show Trotter Circuit",
                             use_container_width=True, key="fh_ckt")

    with right:
        if show_ckt:
            st.subheader("One Trotter Step Circuit")
            dt = t_max / n_steps
            qc_vis = QuantumCircuit(4)
            fh_trotter_step(qc_vis, J, U, dt)
            try:
                st.pyplot(draw_circuit(qc_vis))
            except Exception as e:
                st.error(f"Render error: {e}")

        if run_sim:
            st.subheader("Dynamics")
            with st.spinner(f"Evolving {n_steps} Trotter steps…"):
                try:
                    times, probs = simulate_fh(J, U, t_max, n_steps, init, track)

                    colors = ["royalblue", "tomato", "forestgreen", "purple"]
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for (s, pa), col in zip(probs.items(), colors):
                        ax.plot(times, pa, label=labels.get(s, f"|{s}⟩"),
                                color=col, lw=2)

                    # π markers
                    for k in range(1, int(t_max / np.pi) + 2):
                        xv = k * np.pi
                        if xv <= t_max * 1.01:
                            ax.axvline(xv, color="gray", ls="--",
                                       lw=0.8, alpha=0.5)
                            ax.text(xv, 1.04, f"{k}π",
                                    ha="center", fontsize=8, color="gray")

                    ax.set_xlabel("Time τ  (ħ/J)", fontsize=13)
                    ax.set_ylabel("Probability", fontsize=13)
                    ax.set_title(
                        f"Fermi-Hubbard dynamics  "
                        f"(J={J:.2f},  U={U:.2f},  "
                        f"{n_steps} Trotter steps)",
                        fontsize=13, fontweight="bold",
                    )
                    ax.set_ylim(-0.05, 1.12)
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # ---- Analysis box ----
                    st.markdown("---")
                    st.subheader("Analysis")

                    if init == "1000" and J > 0:
                        pa_site2 = probs.get("0010", np.zeros(1))
                        max_p    = float(np.max(pa_site2))
                        t_max_p  = float(times[np.argmax(pa_site2)])
                        analytic = np.pi / (2 * J)
                        st.metric("Max transfer probability to Site 2",
                                  f"{max_p:.4f}", delta=f"at τ = {t_max_p:.3f}")
                        st.info(
                            f"Analytical Rabi prediction:  "
                            f"ω = 2J = {2*J:.3f},  "
                            f"τ_transfer = π/(2J) = {analytic:.4f}"
                        )

                    elif init == "1100":
                        pa_s1   = probs.get("1100", np.zeros(1))
                        pa_s2   = probs.get("0011", np.zeros(1))
                        max_s2  = float(np.max(pa_s2))
                        ratio   = U / J if J > 0 else float("inf")
                        st.metric("Max doublon tunneling probability", f"{max_s2:.4f}")
                        st.metric("U/J ratio", f"{ratio:.2f}")

                        if ratio > 5:
                            t_eff = J**2 / U if U > 0 else 0
                            st.success(
                                f"🔒 **Mott-insulator regime** (U/J = {ratio:.1f} ≫ 1)  \n"
                                f"Coulomb blockade suppresses doublon tunneling.  \n"
                                f"Effective tunneling rate: J²/U ≈ **{t_eff:.3f}** (vs J = {J:.2f} at U=0).  \n"
                                f"Expected oscillation period: ~**{np.pi/t_eff:.1f}** — "
                                f"increase τ_max to see it."
                            )
                        elif ratio < 1:
                            t_free = np.pi / (2 * J) if J > 0 else 0
                            st.success(
                                f"🔓 **Metallic regime** (U/J = {ratio:.2f} ≪ 1)  \n"
                                f"Free doublon tunneling — peak transfer at τ ≈ **{t_free:.2f}**."
                            )
                        else:
                            st.info(
                                f"⚖️  **Crossover regime** (U/J = {ratio:.2f})  \n"
                                "Competing kinetic and interaction energies."
                            )

                except Exception as e:
                    st.error(f"Simulation error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        if not run_sim and not show_ckt:
            st.info(
                "👈  Set parameters and click **Run Simulation**.  \n\n"
                "**Quick demos:**  \n"
                "- Q3.2 — Non-interacting: U=0, J=1, init=|1000⟩  \n"
                "- Q3.3 — Mott physics: U=10, J=1, init=|1100⟩"
            )


# ============================================================
if __name__ == "__main__":
    main()