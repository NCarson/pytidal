from math import floor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

print(11)


###############################################################################

def _plotHarmonic(ax, harmonic, period,
        title='', 
        use_labels=False, 
        minmax=False, 
        color=None
        ):

    print(22, period.offset)

    ax.clear()
    start, end = period.startEnd()
    time, amplitude = harmonic.xySine(start, end)
    time = period.removeOffset(time)
    ax.plot(period.normalizeTime(time), amplitude, color=color)

    if minmax:
        x, y = harmonic.minmax(time, amplitude)
        x = period.normalizeTime(x)
        ax.plot(x, y, "o", alpha=.3)
        for i, _ in enumerate(x):
            ypad = period.ytextpad(y[i])
            xpad = period.xtextpad(y[i])
            ax.text(x[i] + xpad, y[i] + ypad , round(y[i],2), fontsize=period.fontsize)

    ax.set_xticks(period.xticks)
    ax.set_xlim(*period.xlim)
    if minmax:
        pad = period.ytextpad(1)
        a, b = ax.get_ylim()
        ax.set_ylim(a - abs(pad), b + abs(pad))
    if title:
        ax.set_title(title)
    if use_labels:
        ax.set_xlabel(period.xlabel)
        ax.set_ylabel('Feet')
    ax.grid(True, which='both')

    return ax

def _plotPeaks(ax, harmonic, period, method_name,
        title='', 
        use_labels=False,
        color=None, 
        ):
    print(33, period.offset)

    ax.clear()
    start, end = period.startEnd()
    hour, amplitude = harmonic.xySine(start, end)
    method = getattr(harmonic, method_name)
    x, y = method(hour, amplitude)
    ax.plot(period.normalizeTime(x), y, color=color)

    if title:
        ax.set_title(title)
    if use_labels:
        ax.set_xlabel(period.xlabel)
        ax.set_ylabel('Feet')
    ax.grid(True, which='both')
    return ax

###############################################################################

def plotDay(ax, harmonic, period_kind, offset=0): #FIXME

    f = 24
    ax.clear()
    ax.plot(offset/f, 1, "o", color='black')
    ax.set_title('Day')
    ax.set_yticks([])

    if period_kind == 'day':
        xticks = np.array([0, 1])
    elif period_kind == 'week':
        xticks = np.arange(0, 8)
    elif period_kind == 'month':
        xticks = np.arange(0, 28)
    ax.set_xticks(xticks)


###############################################################################

class AbstractPlotter:

    graphs = None
    height_ratios = None

    @classmethod
    def plot(cls, harmonics, period):

        k = len(cls.graphs)
        grid_spec = GridSpec(k, 1, figure=fig, height_ratios=cls.height_ratios)
        for i, name in enumerate(cls.graphs):
            ax = fig.add_subplot(grid_spec[i, :])
            graph = getattr(cls, name)
            graph(ax, harmonics, period)

    @classmethod
    def plotExpected(cls, ax, harmonic, period):
        _plotHarmonic(ax, harmonic, period, 
            title='Expected Tide', 
            color='darkcyan', 
            minmax=period.length <= 14*24,
            use_labels=True,
            )


class ExpectedPlotter(AbstractPlotter):
    graphs = ['plotExpected']

###############################################################################

class PeakPlotter(AbstractPlotter):

    graphs = [
            'plotExpected',
            'plotHighWater', 
            'plotLowWater', 
    ]

    @classmethod
    def plotHighWater(cls, ax, harmonic, period):
        _plotPeaks(ax, harmonic, period, 'highWater', title='High Water', color='purple')

    @classmethod
    def plotLowWater(cls, ax, harmonic, period):
        _plotPeaks(ax, harmonic, period, 'lowWater', title='Low Water', color='orange')


class MixedPeakPlotter(PeakPlotter):

    graphs = [
            'plotHigherHighWater', 
            'plotLowerHighWater', 
            'plotHigherLowWater', 
            'plotLowerLowWater']

    @classmethod
    def plotHigherHighWater(cls, ax, harmonic, period):
        _plotPeaks(ax, harmonic, period, 'higherHighWater', title='Higher High Water', color='purple')

    @classmethod
    def plotLowerHighWater(cls, ax, harmonic, period):
        _plotPeaks(ax, harmonic, period, 'lowerHighWater', title='High Water', color='blue')

    @classmethod
    def plotHigherLowWater(cls, ax, harmonic, period):
        _plotPeaks(ax, harmonic, period, 'higherLowWater', title='Low Water', color='pink')

    @classmethod
    def plotLowerLowWater(cls, ax, harmonic, period):
        _plotPeaks(ax, harmonic, period, 'lowerLowWater', title='Lower Low Water', color='orange')


###############################################################################

class Kindplotter(AbstractPlotter):

    graphs = [
            'plotExpected', 
            'plotLunar', 
            'plotSolar', 
            'plotWater', 
        ]

    @classmethod
    def plotLunar(cls, ax, harmonic, period):
        _plotHarmonic(ax, harmonic.filterByKind('lunar'), period, title='Lunar', color='y')

    @classmethod
    def plotSolar(cls, ax, harmonic, period):
        _plotHarmonic(ax, harmonic.filterByKind('solar'), period, title='Solar', color='orange')

    @classmethod
    def plotWater(cls, ax, harmonic, period):
        _plotHarmonic(ax, harmonic.filterByKind('water'), period, title='Shallow Water', color='blue')

###############################################################################


class PeriodPlotter(AbstractPlotter):

    graphs = [
            'plotExpected', 
            'plotTerdiurnal', 
            'plotSemidiurnal', 
            'plotDiurnal', 
            'plotAnual']

    @classmethod
    def plotTerdiurnal(cls, ax, harmonic, period):
        _plotHarmonic(ax, harmonic.filterByPeriod('terdiurnal'), period, 
                title='Terdiurnal', color='pink')

    @classmethod
    def plotSemidiurnal(cls, ax, harmonic, period):
        _plotHarmonic(ax, harmonic.filterByPeriod('semidiurnal'), period, 
                title='Semidiurnal', color='purple')

    @classmethod
    def plotDiurnal(cls, ax, harmonic, period):
        _plotHarmonic(ax, harmonic.filterByPeriod('diurnal'), period, 
                title='Diurnal', color='green')

    @classmethod
    def plotAnual(cls, ax, harmonic, period):
        _plotHarmonic(ax, harmonic.filterByPeriod('anual'), period, 
                title='Anual', color='blue')


if __name__ == '__main__':

    import pickle
    from datetime import date

    from api import Station
    from api import HarmonicConstituent
    from api import HarmonicGroup

    from period import HourPeriod
    from period import DayPeriod
    from period import DatePeriod

    with open('data/data.pickle', 'rb') as f:
        station = pickle.load(f)

    all = HarmonicGroup('all', station.harmonics)
    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Tidal Components for Charleston, OR")

    FPS=5
    #anim = FuncAnimation(fig, plotKind, interval=200, frames=4*FPS, blit=False) # interval is in ms
    #anim.save('sine.mp4', writer='ffmpeg', fps=FPS)

    date_start = date(2022, 7, 25)
    date_end = date(2022, 7, 27)
    offset = ((date_start - station.epoch).days -1) * 24
    
    period = DayPeriod(30*24*6)
    period = DayPeriod(6*24, 24*5)
    period = DatePeriod(station.epoch, date_start, date_end)
    period = HourPeriod(24, offset)

    period = HourPeriod(48, -9)

    #Kindplotter.plot(all, period)
    #PeriodPlotter.plot(all, period)
    #MixedPeakPlotter.plot(all, period)
    #PeakPlotter.plot(all.filterByKind('lunar'), period)
    ExpectedPlotter.plot(all, period)

    plt.show()



