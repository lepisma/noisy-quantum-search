import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from collections import namedtuple
from typing import List

State = namedtuple("State", ["index", "coeff"])


def evolve(marking: np.ndarray, states: List[State]) -> List[State]:
    """
    Run one iteration on states

    Parameters:
    -----------
    marking : np.ndarray. shape: (population size,)
    """

    inverted = [State(s.index, marking[s.index] * s.coeff) for s in states]
    mirror = np.mean([i.coeff for i in inverted])

    heads = [State(i.index, i.coeff - mirror) for i in inverted]
    torso = [State(i.index, mirror) for i in inverted]

    return [State(h.index, t.coeff - h.coeff) for t, h in zip(torso, heads)]


def run(markings: np.ndarray) -> np.ndarray:
    """
    Parameters:
    -----------
    markings : np.ndarray. shape: (instances, population size)
    """

    N, P = markings.shape
    init_states = [State(i, 1/np.sqrt(P)) for i in range(P)]

    mem = np.zeros((N + 1, P))
    states = init_states
    mem[0, :] = [s.coeff for s in states]

    for i in range(len(markings)):
        states = evolve(markings[i, :], states)
        # states = evolve(markings[i, :], states)
        mem[i + 1, :] = [s.coeff for s in states]

    return mem

def get_markings(N, P):
    m = np.zeros((N, P))
    return m - 1

def gap_markings(m, gap):
    N = len(m)
    m[:, :3] = 1
    m[N // 5, 0] = -1
    m[(N // 5) + gap, 1] = -1
    return m

def pepper_markings(m, peps, start=10):
    for i in range(len(m)):
        if i <= start:
            # Fill in
            for j, p in enumerate(peps):
                if p > 0:
                    m[i, j] = 1
        else:
            # Pepper
            for j, p in enumerate(peps):
                if np.random.rand() < p:
                    m[i, j] = 1

    return m

def plot_mem(mem: np.ndarray, markings: np.ndarray):
    """
    Parameters:
    -----------
    mem : np.ndarray. shape: (instances + 1, population size)
    markings : np.ndarray. shape: (instances, population size)
    """

    cmap = matplotlib.cm.get_cmap("tab10")
    fitness = ((markings + 1) / 2).mean(axis=0)
    unmarked = [np.isclose(f, 0) for f in fitness]
    colors = [cmap(i) for i in range(mem.shape[1])]

    f, (ax1, ax2) = plt.subplots(2, figsize=(15, 10), sharex=True)

    for p in range(mem.shape[1]):
        if unmarked[p]:
            color = "gray"
            alpha = 0.1
        else:
            color = colors[p]
            alpha = 0.7

        ax1.plot(mem[:, p] ** 2, color=color, alpha=alpha, linewidth=2, label=f"{p} ({fitness[p]})")
        ax2.plot(mem[:, p], color=color, linewidth=2, alpha=alpha, label=f"{p} ({fitness[p]})")

    ax1.set_title("Probability")
    ax2.set_title("Coeff")
    plt.legend(loc="center left")
    plt.grid()
    plt.show()
