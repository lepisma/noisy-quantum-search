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
    """
    Return a set of marking with all unmarked
    """

    m = np.zeros((N, P))
    return m - 1

def mark_all(m, num=3, value=1):
    """
    Mark num elements in the markings
    """

    m[:, :num] = value
    return m

def unmark_one(m, pos, value=-1, row=0):
    """
    Unmark just one element for given pos
    """

    m[pos, row] = value
    return m

def unmark_one_gap(m, gap, start=10, value=-1):
    """
    Unmark two items for one instance with a given gap
    """

    m[start, 0] = value
    m[start + gap, 1] = value
    return m

def unmark_all_noise(m, noise, start=10, value=-1):
    """
    Unmark items for each iteration starting from start using the
    noise vector
    """

    for i in range(start, len(m)):
        for j, p in enumerate(noise):
            if np.random.rand() < p:
                m[i, j] = value
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

    mean_coeff = mem.mean(axis=1)

    f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 15), sharex=True)

    ax1.plot(mean_coeff, linewidth=2, label=f"Mean coeff over genererations")

    for p in range(mem.shape[1]):
        if unmarked[p]:
            color = "gray"
            alpha = 0.1
        else:
            color = colors[p]
            alpha = 0.7

        ax2.plot(mem[:, p] ** 2, color=color, alpha=alpha, linewidth=2, label=f"{p} ({fitness[p]})")
        ax3.plot(mem[:, p], color=color, linewidth=2, alpha=alpha, label=f"{p} ({fitness[p]})")

    ax1.set_title("Mean coefficients")
    ax2.set_title("Probability")
    ax3.set_title("Coeff")
    plt.legend(loc="center left")
    plt.grid()
    plt.show()
