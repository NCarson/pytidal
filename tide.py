from math import floor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
print(11)


class HarmonicSine:

    period_kinds = 'day', 'week', 'month'

    @property
    def xlabel(self):
        k = self._period_kind
        if k == 'day':
            return 'Hour'
        elif k == 'week':
            return 'Day'
        elif k == 'month':
            return 'Day'

    @property
    def xticks(self):
        k = self._period_kind
        if k == 'day':
            return np.array([0, 3, 6, 9, 12, 15, 18, 21, 24]) + self._offset
        elif k == 'week':
            return np.arange(0, 8) + self._offset/24
        elif k == 'month':
            return np.arange(0, 28) + self._offset/24

    @property
    def xlim(self):
        k = self._period_kind
        o = self._offset
        if k == 'day':
            return 0+o, 24+o
        elif k == 'week':
            return (0+o/24, 7+o/24)
        elif k == 'month':
            return (0+o/24, 28+o/24)


    def __init__(self, harmonic, period_kind='day', offset=0):
        if period_kind not in self.period_kinds:
            raise ValueError(f'unknown period_kind {repr(period_kind)}')
        self._harmonic = harmonic
        self._period_kind = period_kind
        self._offset = offset

    def xySine(self, end=24):
        offset = self._offset
        end += offset
        f = 1
        if self._period_kind == 'week':
            end *= 7
            offset *= 7
            f = 24
        elif self._period_kind == 'month':
            end = end * 28
            offset *= 28 
            f = 24

        time, amplitude = self._harmonic.xySine(end=end, offset=self._offset)
        return (time+self._offset)/f, amplitude

    def minmax(self, time, amplitude):
        a = np.diff(np.sign(np.diff(amplitude))).nonzero()[0] + 1 # local min+max

        x, y = np.ones(len(a)), np.ones(len(a))
        for i, k in enumerate(a):
            x[i], y[i] = time[k], amplitude[k]
        return x, y


    def _filter_min_max(self, x, y, cmp):
        x_out, y_out = list(), list()
        last = None
        for i, (xx, yy) in enumerate(zip(x, y)):
            if last and cmp(last, yy):
                x_out.append(xx)
                y_out.append(yy)
            last = yy
        return x_out, y_out

    def _highWater(self, time, amplitude):
        x, y = self.minmax(time, amplitude)
        mask = (y > 0)
        x, y = x[mask], y[mask]
        return x, y

    def highWater(self, time, amplitude):
        return self._highWater(time, amplitude)

    def higherHighWater(self, time, amplitude):
        x, y = self._highWater(time, amplitude)
        return self._filter_min_max(x, y, lambda last,yy : last < yy)

    def lowerHighWater(self, time, amplitude):
        x, y = self._highWater(time, amplitude)
        return self._filter_min_max(x, y, lambda last,yy : last > yy)

    def _lowWater(self, time, amplitude):
        x, y = self.minmax(time, amplitude)
        mask = (y < 0)
        x, y = x[mask], y[mask]
        return x, y

    def lowWater(self, time, amplitude):
        return self._lowWater(time, amplitude)

    def lowerLowWater(self, time, amplitude):
        x, y = self._lowWater(time, amplitude)
        return self._filter_min_max(x, y, lambda last,yy : last > yy)

    def higherLowWater(self, time, amplitude):
        x, y = self._lowWater(time, amplitude)
        return self._filter_min_max(x, y, lambda last,yy : last < yy)

    def moving_average(self, x, n):
        #return np.convolve(x, np.ones(window_size), 'same' ) / window_size
        ret = np.cumsum(x, dtype='float')
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:]/n

###############################################################################

def plotHarmonic(ax, harmonic, 
        offset=0, title='', use_labels=False, minmax=False, 
        color=None, period_kind='day'
        ):
    print(22, offset)

    xlim_pad = .25
    ylim_pad = 0
    ax.clear()
    sine = HarmonicSine(harmonic, period_kind=period_kind, offset=offset)
    time, amplitude = sine.xySine()
    ax.plot(time, amplitude, color=color)

    if minmax:
        ylim_pad = -1
        x, y = sine.minmax(time, amplitude)
        ax.plot(x, y, "o", alpha=.3)
        for i, _ in enumerate(x):
            ax.text(x[i] + (0.1), y[i] + (-1 + 0.01) , round(y[i],1), fontsize=8)

    ax.set_xticks(sine.xticks)
    a, b = sine.xlim
    ax.set_xlim(a, b+xlim_pad)
    a, b = ax.get_ylim()
    ax.set_ylim(a+ylim_pad, b)
    if title:
        ax.set_title(title)
    if use_labels:
        ax.set_xlabel(sine.xlabel)
        ax.set_ylabel('Feet')
    ax.grid(True, which='both')

    return ax

def _plotPeaks(ax, harmonic, method_name,
        offset=0, title='', use_labels=False,
        color=None, period_kind='day'
        ):
    print(33, offset)

    ax.clear()
    sine = HarmonicSine(harmonic, period_kind=period_kind, offset=offset)
    time, amplitude = sine.xySine()
    method = getattr(sine, method_name)
    x, y = method(time, amplitude)
    ax.plot(x, y, color=color)

    if title:
        ax.set_title(title)
    if use_labels:
        ax.set_xlabel(sine.xlabel)
        ax.set_ylabel('Feet')
    ax.grid(True, which='both')
    return ax

###############################################################################

def plotDay(ax, harmonic, period_kind, offset=0):

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
    def plot(cls, harmonics, offset=0, period_kind='day'):

        k = len(cls.graphs)
        grid_spec = GridSpec(k, 1, figure=fig, height_ratios=cls.height_ratios)
        for i, name in enumerate(cls.graphs):
            ax = fig.add_subplot(grid_spec[i, :])
            graph = getattr(cls, name)
            print(11, name, offset)
            graph(ax, harmonics, period_kind, offset=offset)

    @classmethod
    def plotExpected(cls, ax, harmonic, period_kind, offset=0):
        plotHarmonic(ax, harmonic, title='Expected Tide', color='green', 
            minmax=True if period_kind != 'month' else False,
            offset=offset,
            period_kind=period_kind, use_labels=True)


###############################################################################

class PeakPlotter(AbstractPlotter):

    graphs = [
            'plotExpected',
            'plotHighWater', 
            'plotLowWater', 
    ]

    @classmethod
    def plotHighWater(cls, ax, harmonic, period_kind, offset=0):
        _plotPeaks(ax, harmonic, 'highWater', title='High Water', color='purple',
            offset=offset, period_kind=period_kind)

    @classmethod
    def plotLowWater(cls, ax, harmonic, period_kind, offset=0):
        _plotPeaks(ax, harmonic, 'lowWater', title='Low Water', color='orange',
            offset=offset, period_kind=period_kind)

class MixedPeakPlotter(PeakPlotter):

    graphs = [
            'plotHigherHighWater', 
            'plotLowerHighWater', 
            'plotHigherLowWater', 
            'plotLowerLowWater']

    @classmethod
    def plotHigherHighWater(cls, ax, harmonic, period_kind, offset=0):
        _plotPeaks(ax, harmonic, 'higherHighWater', title='Higher High Water', color='purple',
            offset=offset, period_kind=period_kind)

    @classmethod
    def plotLowerHighWater(cls, ax, harmonic, period_kind, offset=0):
        _plotPeaks(ax, harmonic, 'lowerHighWater', title='Lower High Water', color='blue',
            offset=offset, period_kind=period_kind)

    @classmethod
    def plotHigherLowWater(cls, ax, harmonic, period_kind, offset=0):
        _plotPeaks(ax, harmonic, 'higherLowWater', title='Higher Low Water', color='pink',
            offset=offset, period_kind=period_kind)

    @classmethod
    def plotLowerLowWater(cls, ax, harmonic, period_kind, offset=0):
        _plotPeaks(ax, harmonic, 'lowerLowWater', title='Lower Low Water', color='orange',
            offset=offset, period_kind=period_kind)


###############################################################################

class Kindplotter(AbstractPlotter):

    graphs = [
            'plotExpected', 
            'plotLunar', 
            'plotSolar', 
            'plotWater', 
        ]

    @classmethod
    def plotLunar(cls, ax, harmonic, period_kind, offset=0):
        plotHarmonic(ax, harmonic.filterByKind('lunar'), title='Lunar', color='y',
            offset=offset,
            period_kind=period_kind)

    @classmethod
    def plotSolar(cls, ax, harmonic, period_kind, offset=0):
        plotHarmonic(ax, harmonic.filterByKind('solar'), title='Solar', color='orange',
            offset=offset,
            period_kind=period_kind)

    @classmethod
    def plotWater(cls, ax, harmonic, period_kind, offset=0):
        plotHarmonic(ax, harmonic.filterByKind('water'), title='Shallow Water', color='blue',
            offset=offset,
            period_kind=period_kind)

###############################################################################


class PeriodPlotter(AbstractPlotter):

    graphs = [
            'plotExpected', 
            'plotTerdiurnal', 
            'plotSemidiurnal', 
            'plotDiurnal', 
            'plotAnual']

    @classmethod
    def plotTerdiurnal(cls, ax, harmonic, period_kind, offset=0):
        plotHarmonic(ax, harmonic.filterByPeriod('terdiurnal'), title='Terdiurnal', color='pink',
            offset=offset, period_kind=period_kind)

    @classmethod
    def plotSemidiurnal(cls, ax, harmonic, period_kind, offset=0):
        plotHarmonic(ax, harmonic.filterByPeriod('semidiurnal'), title='Semidiurnal', color='purple',
            offset=offset, period_kind=period_kind)

    @classmethod
    def plotDiurnal(cls, ax, harmonic, period_kind, offset=0):
        plotHarmonic(ax, harmonic.filterByPeriod('diurnal'), title='Diurnal', color='green',
            offset=offset, period_kind=period_kind)

    @classmethod
    def plotAnual(cls, ax, harmonic, period_kind, offset=0):
        plotHarmonic(ax, harmonic.filterByPeriod('anual'), title='Anual', color='blue',
            offset=offset, period_kind=period_kind)


###############################################################################


def plotExpectedOnly(harmonics, offset=0, period_kind='month'):

    gs = GridSpec(1, 1, figure=fig, height_ratios=[1])
    graphs = [plotExpected]
    for i, graph in enumerate(graphs):
        ax = fig.add_subplot(gs[i, :])
        graph(ax, harmonics, period_kind, offset=offset)

def plotKind(harmonics, offset=0, period_kind='day'):

    gs = GridSpec(4, 1, figure=fig, height_ratios=[4,2,2,2])
    graphs = [plotExpected, plotLunar, plotSolar, plotWater]
    for i, graph in enumerate(graphs):
        ax = fig.add_subplot(gs[i, :])
        graph(ax, harmonics, period_kind, offset=offset)


if __name__ == '__main__':

    from api import HarmonicConstituent
    from api import HarmonicGroup
    import pickle
    with open('data.pickle', 'rb') as f:
        coefs = pickle.load(f)

    all = HarmonicGroup('all', coefs)
    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Tidal Components for Charleston, OR")

    FPS=5
    #anim = FuncAnimation(fig, plotKind, interval=200, frames=4*FPS, blit=False) # interval is in ms
    #anim.save('sine.mp4', writer='ffmpeg', fps=FPS)

    #MixedPeakPlotter.plot(all)
    #PeriodPlotter.plot(all)
    #Kindplotter.plot(all)
    PeakPlotter.plot(all.filterByKind('lunar'), period_kind='month')
    plt.show()



