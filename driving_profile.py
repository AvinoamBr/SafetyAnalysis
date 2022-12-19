import os

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


# ---- CONSTS --------
OPERATION_START_DT = 2  # [minutes] time delta between signals to be considered new operation,
# -------------------

def dateTime_fix(df, fixes):
    '''
    :param df: data frame with datetime fields that have a known offset
    :param fixes: [(start,end,offset),...]  list of offsets according to time gaps
    :return: fixed data frame
    '''
    for (start,end,offset) in fixes:
        df.datetime[np.logical_and(df.datetime > start, df.datetime < end)] = df.datetime + offset

class VehicleStatus(object):
    def __init__(self, source):
        self.source = source
        # self._signal = signal
        self._signal_df = pd.DataFrame()
        self._events = pd.DataFrame()
        self.grouped_events = pd.DataFrame()
        self._GPS_pd = pd.DataFrame()
        self._load()

    def _load(self):
        self._GPS_pd = pd.read_csv(self.source)
        self._GPS_pd.datetime = pd.to_datetime(self._GPS_pd.datetime, format="%Y%m%d_%H%M%S")
        # GPS_pd = GPS_pd.head(50000)
        dateTime_fix(self._GPS_pd, FIXES)
        self._GPS_pd.index = self._GPS_pd.datetime
        self._filter_by_location(32.6, 32.2, 34.5, 35.5) # (top, bottom, left, right)
        self._signal = self._GPS_pd.speed

    def _filter_by_location(self, top, bottom, left, right):
        t = self._GPS_pd.lat < top
        b = self._GPS_pd.lat > bottom
        l = self._GPS_pd.lon > left
        r = self._GPS_pd.lon < right
        condition = np.all(np.array([t,b,l,r]), axis=0)
        self._GPS_pd = self._GPS_pd[condition]

    def calculte_signal_df(self):
        '''
        use signal data to detect signal existence over period
        tiles over epoch from first to last signal

        signal_stat -> raw data, is there signal for GPS
        smooth_stat -> smoothing signal_stat, to reduce noise affects
        working_stat -> smooth_stat above threshold.
        '''
        start = self._signal.index.min()
        end = self._signal.index.max()
        all_times = pd.date_range(start, end, freq='60s')
        self._signal_df = pd.DataFrame(index=all_times)
        self._signal_df.status = 0
        self._signal_df['signal_stat']= self._signal_df.index.isin(self._signal.index).astype(int)
        self._signal_df['smooth_stat'] =  self._signal_df['signal_stat'].rolling(60,center=True).mean()
        # self._signal_df['working_stat'] = (self._signal_df.smooth_stat>0.5).astype(int)
        self._signal_df['working_stat'] = self._signal_df.signal_stat
        return self._signal_df

    def calculate_events(self):
        '''
        :param signal: pandas series with bollean for signal on/off
        :return: pandas DF of events: [start,end,diration]
        '''
        signal = self._signal_df.working_stat
        df = pd.DataFrame()
        df['signal'] = signal
        diff = df.signal.diff()
        df['diff'] = diff
        df['start'] = np.logical_or(df['diff'] == 1,
                          np.logical_and(pd.isna(df['diff']),df.signal) )
        df['end']= np.logical_or( diff == -1,
                                  np.logical_and(df.signal, df.index == df.index.to_series().iloc[-1]))
        start = (df[df.start]).index
        end = (df[df.end]).index
        events_df = pd.DataFrame(np.array([start, end]).T, columns=['start', 'end'])
        events_df['duration'] = events_df.end - events_df.start
        events_df['date'] = events_df.start.apply(lambda v: datetime.datetime(v.year, v.month, v.day))

        self._events = events_df

    def calculate_rest_time(self):
        speed = self._GPS_pd.speed
        REST_SPEED = 0.5 # minimum working speed
        DRIVE_SPEED = 6  # over this speed - over working speed
        # datetime = speed.index.to_series()
        speed = speed.iloc[np.arange(0, speed.shape[0], 60)]
        df = pd.DataFrame(speed, columns=['speed'])
        # date = speed.index.to_series().apply(lambda v: datetime.date(v.year, v.month, v.day))
        df['datetime'] = df.speed.index.to_series().apply(lambda v: datetime.datetime(v.year, v.month, v.day, v.hour, v.minute))
        df['dt'] = df.index.to_series().diff()
        df['rest'] = (df.speed < REST_SPEED).astype(int)
        # df['operation_start'] = df.dt>datetime.timedelta(minutes=5)
        df['operation_start'] = np.logical_or(df.dt>datetime.timedelta(minutes=OPERATION_START_DT), pd.isna(df.dt))
        df['operation_ends'] = df['operation_start'].shift(-1)

        df['rest_start'] = np.logical_or(df.rest.diff() == 1,
                                         np.logical_and(df.operation_start, df.rest) ).astype(bool)
        df['rest_end'] = np.logical_or(   np.logical_and(df.rest.diff() == -1, df.operation_start==False),
                                          np.logical_and(df.operation_ends, df.rest)  ).astype(bool)
        # df['rest_end'][np.isnan(df.dt)]=False
        self._rest_df = df

        start = df[df.rest_start].index
        end = df[df.rest_end].index
        events_df = pd.DataFrame(np.array([start,end]).T, columns=['start', 'end'])
        events_df['duration'] = events_df.end - events_df.start
        events_df['date'] = events_df.start.apply(lambda v: datetime.datetime(v.year, v.month, v.day))
        self._rest_events = events_df
        pass

    def get_groups_by_day(self, signal='signal', min_duration=None):
        if signal == 'signal':
            events = self._events
        elif signal == 'rest':
            events = self._rest_events
        if min_duration:
            events = events[events.duration>=min_duration]
        self.grouped_events = events.groupby('date')
        groups = self.grouped_events
        return groups


    def get_on_status_index(self):
        return self._signal_df[self._signal_df.working_stat==1].index

    def get_rest_status_index(self):
        t_range = pd.Series()
        events = self._rest_events
        events = events[events.duration> datetime.timedelta(minutes=10)]
        for i,v in events.iterrows():
            t_range = pd.concat( (t_range,pd.date_range(v.start, v.end, freq='10s').to_series()) )
        return t_range

    def plot_working_hours(self):
        signal_by_date = self.get_groups_by_day(signal='signal', min_duration=datetime.timedelta(minutes=10))
        signal_by_date_sum = signal_by_date.sum()
        rest_by_date = self.get_groups_by_day(signal='rest', min_duration=datetime.timedelta(minutes=10))
        rest_by_date_sum = rest_by_date.sum()
        work = signal_by_date_sum
        rest = rest_by_date_sum
        work_rest = pd.concat((work,rest),axis=1)
        work_rest.columns = ['work','rest']
        if rest.shape[0]:
            work_rest.rest.fillna(datetime.timedelta(0))
            work_rest.work[work_rest.work>work_rest.rest] = work_rest.work-work_rest.rest
        # work = work_rest.work
        fig = plt.figure(figsize=(20, 10))
        # ax = plt.subplot(1, 2, 1)
        plt.bar(work.index.to_series(), work.duration.apply(lambda td: td.seconds / 3600),
                width=0.5, label='work')

        plt.bar(rest_by_date_sum.index.to_series(), rest_by_date_sum.duration.apply(lambda td: td.seconds / 3600),
                bottom=work.duration[work.index.isin(rest_by_date_sum.index)].apply(lambda td: td.seconds / 3600),
                width=0.5, label='rest')
        vehicle_id = self.source.split(os.sep)[-2]
        vehicle_type = VEHICLES[vehicle_id]
        plt.title(f"vehicle: {vehicle_type}")
        plt.legend()
        plt.xticks(work.index.to_series(),
                   work.index.to_series().apply(lambda d: f"{d.day}/{d.month}"),
                   rotation=75, fontsize=8)
        plt.ylabel("Working hours")

        # ax = plt.subplot(1,2,2)
        # on = self.get_on_status_index()
        # rest = self.get_rest_status_index()
        # on = on[np.logical_not(on.isin(rest))]
        # plt.scatter(self._GPS_pd.datetime, self._GPS_pd.speed.rolling(60, center=True).mean(), s=1, c='k')
        # plt.scatter(on, [-0.02] * on.shape[0], s=1, c='b')
        # plt.scatter(rest, [-0.015] * rest.shape[0], s=2, c='orange')
        # myFmt = mdates.DateFormatter('%d/%m-%H:%M')
        # plt.xticks([],[],
        #            rotation=75, fontsize=8)
        # ax.xaxis.set_major_formatter(myFmt)
        # plt.title(f"velocity along time")

        # plt.show()
        fn = f"/home/ception/Desktop/NetiveiIsrael/Images/DriveProfile/{vehicle_type}.png"
        plt.savefig(fn)
        print (f"saved figure {fn}")



if __name__ == '__main__':
    for source in sources:
        vs = VehicleStatus(source)
        vs.calculte_signal_df()
        vs.calculate_events()
        vs.calculate_rest_time()
        vs.plot_working_hours()

    exit()



    on = vs.get_on_status_index()
    rest = vs.get_rest_status_index()

    # GPS_pd.speed = GPS_pd.speed.rolling(30).mean()
    speed = GPS_pd.speed
    lat = GPS_pd.lat
    lon = GPS_pd.lon
    plt.figure(figsize=(10,20))
    plt.title(GPS_csv_roller_1.split(os.sep)[-2])
    # plt.subplot(2,2,1)
    # plt.plot(GPS_pd.datetime,GPS_pd.speed)
    # plt.plot(GPS_pd.datetime,GPS_pd.speed.rolling(60, win_type='gaussian').mean(std=3))
    plt.scatter(GPS_pd.datetime,GPS_pd.speed.rolling(60, center=True).mean(), s=1)
    plt.scatter(on, [1]*on.shape[0], s=1)
    plt.scatter(rest, [0.5]*rest.shape[0], s=2, c='y')
    plt.scatter(rest, [0.5]*rest.shape[0], s=2, c='y')

    plt.xticks(rotation=45)
    # plt.subplot(2,2,2)
    # plt.scatter(lat,lon,c=speed, s=1)
    # plt.subplot(2,2,3)
    # plt.plot(GPS_pd.datetime,lat)
    # plt.subplot(2,2,4)
    # plt.plot(GPS_pd.datetime,lon)
    plt.show()
    pass

