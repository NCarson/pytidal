from datetime import date
from datetime import datetime
from collections import OrderedDict

import requests
import numpy as np
import xmltodict
import json

'''
This is the help page to figure out how to use the api correctly:
https://www.tidesandcurrents.noaa.gov/api-helper/url-generator.html

## To Get Accurate Predictions You Must Know That:

- Time is in hours starting from the first epoch date (from the 'datums' api - someone did not go to cowedge). At time of this writing this seems to be 1983-1-1. Buttt, they say they are in process of updating this as I write.
- Amplitudes can be given in feet or meters (check the api help page.)
- Speeds are given in degrees and need to be converted to radians for calculations.
- Phases are given in degrees and need to be converted to radians for calculations.
- Phases are given in local (with daylight savings?) and gmt, BUT you still have to account it with an offset when doing calcations. WHY? #FIXME
- A verticle intercept must be included `a cos(0)` as the harmonics are centered around zero. At the the time of writing this seems to be Lower Low Water LLW (find on the 'datums' api), but more stations need to be checked.
- Your first check should be against the first epoch day. Load up the station in NOAA tides and see if its close. You may need to account for timezone or verticle intercept differences. If you cant get the first day right, your not going to get any others.
- I would also set up everything in GMT at first just to get rid of a confounding factor.

https://tidesandcurrents.noaa.gov/noaatidepredictions.html?id={YOUR STATIONID HERE}&units=standard&bdate=19830101&edate=19830102&timezone=LST&clock=12hour&datum=MLLW&interval=hilo&action=dailychart
'''

class ApiError(Exception): pass

class AbstractDataGetter:

    product = None
    result_name = None
    url = ('https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?'
            + 'begin_date={}'
            + '&end_date={}'
            + '&station={}'
            + '&product={}'
            + '&datum=STND&time_zone=gmt&units=english&format=json'
        )

    @classmethod
    def fromID(cls, station_id, start, end):
        s = start.strftime('%Y%m%d')
        e = end.strftime('%Y%m%d')
        url = cls.url.format(s, e, station_id, cls.product)
        r = requests.get(url)
        j = json.loads(r.text)
        if j.get('error'):
            raise ApiError(j)
        return cls(station_id, start, end, j[cls.result_name])

    def __init__(self, station_id, start, end, j):
        self._station_id = station_id
        self._start = start
        self._end = end
        self._times = [] 
        self._values = [] 
        for item in j:
            time = item['t']
            time = datetime.strptime(time, '%Y-%m-%d %H:%M')
            self._times.append(time)
            value = float(item['v'])
            self._values.append(value)

    @property
    def station_id(self): return self._station_id
    @property
    def times(self): return self._times
    @property
    def values(self): return self._values
    @property
    def start(self): return self._start
    @property
    def end(self): return self._end


class WaterLevels(AbstractDataGetter):
    product = 'water_level'
    result_name = 'data'


class Predictions(AbstractDataGetter):
    product = 'predictions'
    result_name = 'predictions'



class Station:

    url = 'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{}.json?units=english'

    @classmethod
    def fromID(cls, id):
        r = requests.get(cls.url.format(id))
        return cls(json.loads(r.text))

    def __init__(self, j):

        station = j['stations'][0]
        self.id = station['id']
        self.name = station['name']
        self.lat = station['lat']
        self.lng = station['lng']
        self.type_type = station['tideType']
        self.state = station['state']
        self.timezone = station['timezone']
        self.timezonecorr = station['timezonecorr']

        self.harmonics = OrderedDict()
        for harmonic in HarmonicConstituent.fromID(self.id):
            self.harmonics[harmonic.name] = harmonic

        datums = StationDatums.fromID(self.id)
        self.epoch = datums.epoch
        self.datums = datums.datums
        llw = self.datums['MLLW'] #TODO check if this right
        self.setZ0(llw)

    def __repr__(self):
        n = self.__class__.__name__
        return f'<{n} #{self.id} {self.name}, {self.state}>'

    def setZ0(self, value):
        d = {
                'number': 99,
                'name': 'Z0',
                'description': 'vertical intercept',
                'amplitude': value,
                'phase_GMT': 0,
                'phase_local': 0,
                'speed': 180,
        }
        self.harmonics['Z0'] = (HarmonicConstituent('feet', d)) #FIXME feet or meter


class StationDatums:
    url = 'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{}/datums.xml?units=english'
    @classmethod
    def fromID(cls, id):
        r = requests.get(cls.url.format(id))
        d = xmltodict.parse(r.text)
        return cls(d)

    def __init__(self, j):

        epoch = j['Datums']['epoch']
        epoch = int(epoch.split('-')[0])
        self._epoch = date(epoch, 1, 1)
        self._datums = {}
        for d in j['Datums']['Datum']:
            self._datums[d['name']] = float(d['value'])

    @property
    def epoch(self): return self._epoch

    @property
    def datums(self): return self._datums


class HarmonicConstituent:

    url = 'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{}/harcon.json?units=english'
    period_kinds = set(['terdiurnal', 'semidiurnal', 'diurnal', 'anual'])
    kinds = set(['solar', 'lunar', 'water'])

    @classmethod
    def fromID(cls, id):
        r = requests.get(cls.url.format(id))
        j = json.loads(r.text)
        units = j['units']
        harmonics = []
        for item in j['HarmonicConstituents']:
            harmonic = cls(units, item)
            harmonics.append(harmonic)
        return harmonics

    def __init__(self, units, j):

        self._units = units
        self._number = j['number']
        self._name = j['name']
        self._description = j['description']
        self._amplitude = j['amplitude']
        self._phase_GMT= j['phase_GMT']
        self._phase_local= j['phase_local']
        self._speed = j['speed']

    def __repr__(self):
        n = self.__class__.__name__
        return f'<{n} {self.number}:{self.name} {self.amplitude}>'
    
    @property
    def units(self): return int(self._units)
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


    def xySine(self, start, end, inc=.1, datum=0, use_local=False):
        '''

This is explained in NOAA's Tidal Analysis and Predictions, page 90
https://tidesandcurrents.noaa.gov/publications/Tidal_Analysis_and_Predictions.pdf

`h(t) = H_o + Sum(f_i H_i cos(a_i t + {V_o + u}_i - k_i))`

* h(t) = Height of tide at any time t.

* H_o = Mean height of water level above the datum.  (like mean lower low water from station datum page)

* f_i = factor for reducing mean amplitude H_i to year of prediction. (??? 18.6 year lunar nodal cycle)

* H_i = amplitude of tidal consitiuent i (amplitude from harmonics page. in feet or meters)

* a_i = speed of tidal constituent i. (speed from harmonics page. in degrees/hour)

* t = time reckoned from some initial epoch such as beginning of year of predictions. (in hours)

* {V_o+u}_i = equilibrium argument for tidal constituent i at t=0. (??? in degrees)

* k_i = epoch (phase lag) of tidal constituent i relative to the moon's transit over the tide station.  (in degress) !! The units for K (Local Standard Time or GMT) will determine the timezone of the times of the tide predictions generated.

A simplier but more confusing version is available here:
https://tidesandcurrents.noaa.gov/about_harmonic_constituents.html
        '''

        from datetime import date
        k = (date(1983, 1, 1) - date(1970, 1, 1)).days * 24

        hour = np.arange(start, end, inc);
        s = self._toRad(self.speed)
        if use_local:
            p = self._toRad(self.phase_local)
        else:
            p = self._toRad(self.phase_GMT)
        x = np.arange(start, end, inc) + k
        amplitude = datum + self.amplitude * np.cos(s * x + p - k)
        return hour, amplitude

    def minmax(self, hour, amplitude):
        a = np.diff(np.sign(np.diff(amplitude))).nonzero()[0] + 1 # local min+max

        x, y = np.ones(len(a)), np.ones(len(a))
        for i, k in enumerate(a):
            x[i], y[i] = hour[k], amplitude[k]
        return x, y

    def _filter_min_max(self, x, y, cmp):
        x_out, y_out = list(), list()
        last = None
        for i, (xx, yy) in enumerate(zip(x, y)):
            if last and cmp(last, yy):
                x_out.append(xx)
                y_out.append(yy)
            last = yy
        return np.array(x_out), np.array(y_out)

    def _highWater(self, hour, amplitude):
        x, y = self.minmax(hour, amplitude)
        mask = (y > 0)
        x, y = x[mask], y[mask]
        return x, y

    def highWater(self, hour, amplitude):
        return self._highWater(hour, amplitude)

    def higherHighWater(self, hour, amplitude):
        x, y = self._highWater(hour, amplitude)
        return self._filter_min_max(x, y, lambda last,yy : last < yy)

    def lowerHighWater(self, hour, amplitude):
        x, y = self._highWater(hour, amplitude)
        return self._filter_min_max(x, y, lambda last,yy : last > yy)

    def _lowWater(self, hour, amplitude):
        x, y = self.minmax(hour, amplitude)
        mask = (y < 0)
        x, y = x[mask], y[mask]
        return x, y

    def lowWater(self, hour, amplitude):
        return self._lowWater(hour, amplitude)

    def lowerLowWater(self, hour, amplitude):
        x, y = self._lowWater(hour, amplitude)
        return self._filter_min_max(x, y, lambda last,yy : last > yy)

    def higherLowWater(self, hour, amplitude):
        x, y = self._lowWater(hour, amplitude)
        return self._filter_min_max(x, y, lambda last,yy : last < yy)

    def moving_average(self, x, n):
        #return np.convolve(x, np.ones(window_size), 'same' ) / window_size
        ret = np.cumsum(x, dtype='float')
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:]/n


class HarmonicGroup(HarmonicConstituent):
    def __init__(self, name, harmonics=[], description=''):
        self._name = name
        self._harmonics = list(harmonics)
        self._description = description

    def __getitem__(self, key):
        for h in self._harmonics:
            if h.name == key:
                return h
        raise KeyError(key)

    @property
    def name(self): return self._name
    @property
    def harmonics(self): return list(self._harmonics)
    @property
    def description(self): return self._description

    def __repr__(self):
        n = self.__class__.__name__
        return f'<{n} {self.name}>'
    
    def append(self, item):
        self._harmonics.append(item)

    def __iter__(self): return iter(self._harmonics)

    def xySine(self, start, end, inc=.1):
        if not self._harmonics:
            raise ValueError('harmonics are empty')
        l = []
        for h in self._harmonics:
            hour, amplitude = h.xySine(start, end, inc=inc)
            l.append(amplitude)
        return hour, np.sum(l, axis=0)

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


if __name__ == '__main__':

    id = 9432780
    s, e = date(2022, 7, 26), date(2022, 7, 27)
    data = WaterLevels.fromID(id, s, e)
    print(11, data.times)
    print(11, data.values)
    '''
    station = Station.fromID(id)
    print(station)
    for i, key in enumerate(station.harmonics):
        print(i, key)

    import pickle

    with open('data/data.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(station, f, pickle.HIGHEST_PROTOCOL)
    '''


