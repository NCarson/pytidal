import requests
import xml.etree.ElementTree as ET
import numpy as np

def hourMins(hours):
    return floor(hours), round((hours - floor(hours)) * 60)
#h, m = hourMins(360 / speed)

class HarmonicConstituent:

    url = 'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{}/harcon.xml'
    period_kinds = set(['terdiurnal', 'semidiurnal', 'diurnal', 'anual'])
    kinds = set(['solar', 'lunar', 'water'])

    def __init__(self, item, station_id=None):
        self.station_id = station_id
        for x in item:
            setattr(self, '_' + x.tag, x.text)

    def __repr__(self):
        n = self.__class__.__name__
        return f'<{n} {self.number}:{self.name}>'
    
    @classmethod
    def fromID(cls, id):
        r = requests.get(cls.url.format(id))
        root = ET.fromstring(r.text)
        units = None
        coefs = []
        for i, item in enumerate(root):
            if not i:
                units = item.text
            else:
                coef = HarmonicConstituent(item, station_id=id)
                if coef.amplitude:
                    coefs.append(coef)
        return coefs

    @property
    def number(self): return int(self._number)
    @property
    def name(self): return self._name
    @property
    def description(self): return self._description
    @property
    def amplitude(self): return float(self._amplitude)
    @property
    def phase_GMT(self): return float(self._phase_GMT)
    @property
    def phase_local(self): return float(self._phase_local)
    @property
    def speed(self): return float(self._speed)
    @property
    def period(self): return 360 / float(self._speed)
    @property
    def kind(self):
        d = self.description.lower()
        if 'water' in d:
            return 'water'
        elif 'solar' in d:
            return 'solar'
        elif 'lunar' in d:
            return 'lunar'

    @property
    def period_kind(self):
        p = self.period
        if p < 12:
            return 'terdiurnal'
        elif p <= 14:
            return 'semidiurnal'
        elif p < 28:
            return 'diurnal'
        else:
            return 'anual'
            
    def _toRad(self, degree):
        return degree * (np.pi / 180)

    def xySine(self, start=0, end=24, offset=0, inc=.1):
        hour = np.arange(start, end, inc);
        s = self._toRad(self.speed)
        p = self._toRad(self.phase_local)
        amplitude = self.amplitude * np.sin(s * (hour+offset) + p)
        return hour, amplitude

class HarmonicGroup:
    def __init__(self, name, harmonics=[], description=''):
        self._name = name
        self._harmonics = list(harmonics)
        self._description = description

    def filterByKind(self, kind, description=''):
        l = []
        for h in self:
            if kind not in h.kinds:
                raise ValueError(f'Unknown kind {repr(kind)}')
            if h.kind == kind:
                l.append(h)
        return self.__class__(kind, l, description)

    def filterByPeriod(self, kind, description=''):
        l = []
        for h in self:
            if kind not in h.period_kinds:
                raise ValueError(f'Unknown period_kind {repr(kind)}')
            if h.period_kind == kind:
                l.append(h)
        return self.__class__(kind, l, description)

    @property
    def name(self): return self._name
    @property
    def harmonics(self): return list(self._harmonics)
    @property
    def description(self): return self._description

    def __repr__(self):
        n = self.__class__.__name__
        return f'<{n} {self.name}>'

    def __iter__(self): return iter(self._harmonics)
    
    def append(self, item):
        self._harmonics.append(item)

    def xySine(self, start=0, end=24, offset=0, inc=.1):
        if not self._harmonics:
            raise ValueError('harmonics are empty')
        l = []
        for h in self._harmonics:
            hour, amplitude = h.xySine(start, end, offset, inc)
            l.append(amplitude)

        return hour, np.sum(l, axis=0)


if __name__ == '__main__':

    coefs = HarmonicConstituent.fromID(8665530)
    import pickle

    with open('data.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(coefs, f, pickle.HIGHEST_PROTOCOL)

