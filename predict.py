#system
from datetime import datetime
from datetime import timedelta
import pickle

# 3rd party
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pytides.constituent as cons
from pytides.tide import Tide

# local
from api import Station
from api import HarmonicConstituent

with open('data/data.pickle', 'rb') as f:
    station = pickle.load(f)

station.harmonics.pop('Z0')
published_amplitudes = [c.amplitude for c in station.harmonics.values()]
published_phases = [c.phase_GMT for c in station.harmonics.values()]

#We can add a constant offset (e.g. for a different datum, we will use relative to MLLW):
#MTL = 5.113
#MLLW = 3.928
#offset = MTL - MLLW
offset = station.datums['MTL']
constituents = [c for c in cons.noaa if c != cons._Z0]
constituents.append(cons._Z0)
published_phases.append(0)
published_amplitudes.append(offset)

#Build the model.
assert(len(constituents) == len(published_phases) == len(published_amplitudes))
model = np.zeros(len(constituents), dtype = Tide.dtype)
model['constituent'] = constituents
model['amplitude'] = published_amplitudes
model['phase'] = published_phases


if __name__ == '__main__':

    from datetime import date
    from api import Predictions
    from api import WaterLevels

    def plot(ax, x, y, color=None, linewidth=None, alpha=None):

        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha)
        ax.grid(True, which='both')
        return ax
    
    def saveData():
        id = 9432780
        delta = timedelta(days=1)
        p = Predictions.fromID(id, start, end-delta)
        wl = WaterLevels.fromID(id, start, end-delta)

        with open('data/predict.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump([p, wl], f, pickle.HIGHEST_PROTOCOL)

    def getData():
        with open('data/predict.pickle', 'rb') as f:
            p, wl = pickle.load(f)
            print(p)
            print(wl)
        return p, wl

    start, end = datetime(1983,1,1), datetime(1983,1,2)
    start, end = datetime(2021,7,1), datetime(2021,7,2)
    start, end = datetime(2022,7,1), datetime(2022,7,2)

    saveData()
    p, wl  = getData()

    fig = plt.figure(constrained_layout=True)
    grid_spec = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(grid_spec[0:])

    plot(ax, wl.times, wl.values, color='black')

    tide = Tide(model = model, radians = False)
    x = np.arange(start, end, timedelta(minutes=6)).astype(datetime)
    y = tide.at(x)
    plot(ax, x, y, color='pink')

    plot(ax, p.times, p.values, color='green', linewidth=1, alpha=.5)
    ax.legend(['observed', 'pytide', 'noaa', ])

    plt.show()

