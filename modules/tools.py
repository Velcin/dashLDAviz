import numpy as np
import math

def rescale(d, p):
    _min = np.min(list(d.values()))
    _max = np.max(list(d.values()))
    norm = {k:math.floor((v - _min)*p/(_max-_min)) for k,v in d.items()}
    #for k, v in norm.items():
    #	if v>p:
    #		print("min {} et max {}, valeur ancienne {} et nouvelle {}".format(_min, _max, d[k], norm[k]))
    return norm

# courtesy to tklab-tud
# License: MIT License
# found on: https://www.programcreek.com/python/?CodeExample=calculate+entropy
def calculate_entropy(frequency: list, normalized: bool = False):
        """
        Calculates entropy and normalized entropy of list of elements that have specific frequency
        :param frequency: The frequency of the elements.
        :param normalized: Calculate normalized entropy
        :return: entropy or (entropy, normalized entropy)
        """
        entropy, normalized_ent, n = 0, 0, 0
        sum_freq = sum(frequency)
        for i, x in enumerate(frequency):
            p_x = float(frequency[i] / sum_freq)
            if p_x > 0:
                n += 1
                entropy += - p_x * math.log(p_x, 2)
        if normalized:
            if math.log(n) > 0:
                normalized_ent = entropy / math.log(n, 2)
            return entropy, normalized_ent
        else:
            return entropy