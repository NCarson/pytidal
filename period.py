import numpy as np

class AbstractPeriod:

    xlabel = None
    ytextpad_up = None
    ytextpad_down = None
    xtextpad_left = None
    fontsize = None

    def __init__(self, length, offset):
        self._offset = offset
        self._length = length

    @property
    def offset(self): return self._offset
    @property
    def length(self): return self._length

    def startEnd(self):
        o, l = self._offset, self._length
        return o, o + l

    def xlim(self):
        raise NotImplimentedError()

    def xticks(self):
        raise NotImplimentedError()

    def ytextpad(self, amp):
        if amp > 0:
            return self.ytextpad_up
        return self.ytextpad_down

    def xtextpad(self, amp):
        return self.xtextpad_left

class HourPeriod(AbstractPeriod):

    xlabel = 'Hour'

    ytextpad_up = .5
    ytextpad_down = -.75
    xtextpad_left = -.5
    fontsize = 12

    def __init__(self, length=24, offset=0):
        super().__init__(length, offset)

    def normalizeTime(self, hour):
        return hour

    def removeOffset(self, hour):
        return hour - (self._offset // 24) * 24

    @property
    def xticks(self):
        o, l = self._offset, self._length
        o = o - o // 24 * 24
        x = np.arange(o, o+l+3, 3)
        return x

    @property
    def xlim(self):
        o, l = self._offset, self._length
        o = o - o // 24 * 24
        return o, l + o


class DayPeriod(AbstractPeriod):

    xlabel = 'Day'
    ytextpad_up = .5
    ytextpad_down = -.75
    xtextpad_left = -.2
    fontsize = 8

    def __init__(self, length=24*7, offset=0):
        super().__init__(length, offset)

    def normalizeTime(self, hour):
        return hour / 24

    def removeOffset(self, hour):
        return hour - (self._offset // 1)

    @property
    def xticks(self):
        o, l = self._offset / 24 , self._length / 24
        o = o - o // 1
        x = np.arange(o, o+l+1)
        return x

    @property
    def xlim(self):
        o, l = self._offset / 24, self._length / 24
        o = o - o // 1
        return o, o+l


class DatePeriod(DayPeriod):

    def __init__(self, epoch, date_start, date_end):
        self._epoch = epoch
        self._date_start = date_start
        self._date_end = date_end

        offset = (self._date_start - self._epoch).days * 24
        length = (date_end - date_start).days * 24
        super().__init__(length, offset)


