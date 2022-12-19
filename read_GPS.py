import glob
import logging
import os
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pynmea2
from matplotlib import dates

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ----------------- Helper methods to extract absolute time from files---------------
def convert_from_ms(milliseconds):
    seconds, milliseconds = divmod(milliseconds,1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    microsecond = milliseconds*1000
    return days, hours, minutes, seconds, microsecond

def datetime_from_file_path(file_path):
    date_str = re.findall(r"\/\d{8}\/",file_path)[-1].replace("/","")
    return datetime.strptime(date_str,"%Y%m%d")
# ----------------- /Helper methods -------------------------------------------------


class GPS_Data(object):
    def __init__(self, site, *args, **kwargs):
        self.worked = pd.DataFrame()
        self.speed_array = np.array([]) # place holder for : np.array n*2: [[date_time, speed],...]
        self.site = site

    def _read_GPS_file(self, fn):
        '''
        read a single ception GPS file
        parse all $GPVTG messages using pynmea2 parser

        :return
         np.array n*2: [[date_time, speed],...]
        '''

        gps_data = []
        date = datetime_from_file_path(fn)
        year,month,day = date.year,date.month,date.day
        with open(fn,'r') as f:
            for l in f.readlines():
                try:
                    # l = f.readline()
                    ms_from_midnight = int(l.split(",")[0])
                    _, hours, minutes, seconds, microseconds = convert_from_ms(ms_from_midnight)
                    date_time = datetime(year=year, month=month, day=day, hour=hours, minute=minutes,
                                         second=seconds, microsecond=microseconds)
                    ll = ",".join(l.split(",")[1:])
                    parsed = pynmea2.parse(ll)
                    if ll.startswith("$GPVTG") : #and parsed.spd_over_grnd_kmph>0:
                        speed_value = parsed.spd_over_grnd_kmph
                        if speed_value == None:
                            gps_data.append((date_time,None))
                            continue
                        speed = float(speed_value)
                        gps_data.append((date_time,speed))

                except Exception as e:
                    continue
        speed_array = np.array(gps_data)
        return speed_array

    def read_GPS_files_from_folder(self, path):
        GPS_files = glob.glob(path, recursive=True)
        GPS_files = sorted(GPS_files)

        speed_array = np.array([[None, None]])
        numfiles = len(GPS_files)
        for (i,fn) in  enumerate(sorted(GPS_files)):
            datetime_from_file_path(fn)
            logging.info(f"{i}/{numfiles} read file {fn}")
            try:
                speeds = self._read_GPS_file(fn)
            except Exception as e:
                continue
            if speeds.shape[0] != 0:
                speed_array = np.row_stack((speed_array, speeds))

        self.speed_array = speed_array
        self.gps_df = pd.DataFrame(self.speed_array, columns=['datetime', 'speed'])

    def save(self, fn):
        np.save(fn, self.speed_array)

    def load(self, fn):
        self.speed_array = np.load(fn, allow_pickle=True)
        self.gps_df = pd.DataFrame(self.speed_array, columns=['datetime', 'speed'])

    def analyse_work_stop_events(self):
        '''
        FIRST - extract from speed data all STOP events.
            any events starts were vehicle stopped moving, and ends where it started again
        THEN ANALYSE:
            short stops are not tagged, as this is the regular manner of working
            stops that are longer than PAUSE_EVENT_MIN_DURATION are considered as 'pause event'
            stops that are pause_events AND have a missing signal (engine is off) for more than MIN_TIME_FOR_NON_CONTINOUS_EVENT
                are considered 'NOT_CONTINOUS'
        '''

        PAUSE_EVENT_MIN_DURATION = 60 # [sec]
        MIN_TIME_FOR_NON_CONTINOUS_EVENT = 120 # [sec]

        def is_event_continous(event, gps_df):
            events_gps = gps_df[np.logical_and(gps_df.datetime >= event.ends, gps_df.datetime <= event.starts)]
            if events_gps.shape[0]==1:
                return True
            if not (event.pause_event):
                return True
            return np.max(events_gps.datetime.diff()) <= timedelta(seconds=MIN_TIME_FOR_NON_CONTINOUS_EVENT)

        gps_array = self.speed_array
        speed = self.speed_array.T[1]
        zero_velocity = (speed==0).astype(int)
        stopped_ids = np.where(np.diff(zero_velocity,prepend=0)==1)[0] # vehicle stopped event is on
        started_ids = np.where(np.diff(zero_velocity, append=0)==-1)[0] # vehicle started, event is off
        starts = gps_array.T[0][started_ids] # started to drive with speed>0 (end of event)
        ends = gps_array.T[0][stopped_ids] # stopped to drive, velocity ==0 (begin of event)
        duration = starts-ends
        pause_event = duration>timedelta(seconds=PAUSE_EVENT_MIN_DURATION) # event whose stop is considered pause in operation
        self.events = pd.DataFrame(np.column_stack([stopped_ids,started_ids,starts,ends,duration,pause_event]),
                              columns = ['stopped_ids', 'started_ids', 'starts', 'ends', 'duration','pause_event'])
        continous_events = []
        # self.gps_df = pd.DataFrame(gps_array, columns=['datetime', 'speed'])
        numevents = events.shape[0]
        for i,e in self.events.iterrows():
            continous = is_event_continous(e,  self.gps_df) # event is considered continous if signal is not stopping for more that one minute.
            print(f"{i}/{numevents} duration: {e.duration}, continous: {continous}")
            continous_events.append(continous)
        self.events['continous_events'] = continous_events

        return self.gps_df, self.events

    def save_work_stop_events(self, fn):
        self.events.to_csv(fn)

    def load_work_stop_events(self, fn):
        self.events = pd.read_csv(fn)

    def tag_frame_status(self):
        '''
        using the stop events, tag ANY_FRAME if is
            working (velocity>0 or short stops)
            pause (stops with engine on)
            off (stops, before or after engine power-off

        :return:
        '''

        self.worked = pd.Series(["worked"] * self.gps_df.shape[0])
        for (i, e) in self.events.iterrows():
            if not e.continous_events:
                self.worked[e.stopped_ids:e.started_ids] = 'off'
            elif e.pause_event:
                self.worked[e.stopped_ids:e.started_ids] = 'pause'

    def save_frame_status(self, fn):
        self.worked.to_csv(fn, index=False)

    def load_frame_status(self, fn):
        self.worked = pd.read_csv(fn, names=['worked']).worked

    def plot_frame_status(self):
        gps_df = self.gps_df
        for (work_status,color) in [('worked','g'), ('off','r'), ('pause','y')]:
            speed = (gps_df[self.worked == work_status]).speed
            d = gps_df[self.worked == work_status].datetime
            plt.scatter(d, speed, label=work_status, s=1, color=color)
        plt.scatter(self.speed_array.T[0][self.speed_array.T[1] == None], np.zeros_like(self.speed_array.T[0][self.speed_array.T[1] == None]),
                    label="missing_data",s=1)
        plt.legend()
        plt.title(self.site)
        plt.show()

    def count_by_day(self):
        worked = self.worked
        self.gps_df['date'] = self.gps_df['datetime'].apply(lambda dt: dt.date())
        worked_by_day = self.gps_df[worked == 'worked']['date'].value_counts().sort_index()
        pause_by_day = self.gps_df[worked == 'pause']['date'].value_counts().sort_index()
        off_by_day = self.gps_df[worked == 'off']['date'].value_counts().sort_index()

        start_date = min((worked_by_day.index[0], pause_by_day.index[0], off_by_day.index[0]))
        end_date = max((worked_by_day.index[-1], pause_by_day.index[-1], off_by_day.index[-1]))

        all_dates = np.arange(start_date, end_date)
        all_dates = np.append(all_dates, end_date)

        # add zeros were there is missing data, to enable bars bottom
        for work_status in [worked_by_day, pause_by_day, off_by_day]:
            for d in all_dates:
                if d not in work_status.index.values:
                    work_status.loc[d] = 0

        self.pause_by_day = pause_by_day.sort_index()
        self.worked_by_day = worked_by_day.sort_index()
        self.off_by_day = off_by_day.sort_index()

    def plot_by_day(self):
        plt.bar(x=self.worked_by_day.index, height=self.worked_by_day, label='work',
                color='g')
        plt.bar(x=self.pause_by_day.index, height=self.pause_by_day, label='pause',
                bottom=self.worked_by_day, color='y')
        plt.bar(x=self.off_by_day.index, height=self.off_by_day, label='parking',
                bottom=self.pause_by_day + self.worked_by_day, color='r')
        plt.legend()
        plt.title(self.site)
        plt.show()



if __name__ == "__main__":
    site = "Yavne" # not ready
    # site = "Golani"
    # site = "Modiim"
    # site = "Mesubim" # not ready


    path = f"/media/performance/Data/Cemex/Readymix/{site}/*/GPS/2022*/**"
    GPS_files_path = f"{path}/NMEARecorder_2022*"
    os.makedirs(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}",exist_ok=True)
    gps_data = GPS_Data(site)

    # gps_data.read_GPS_files_from_folder(GPS_files_path)
    # gps_data.save(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/speed.npy")
    gps_data.load(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/speed.npy")
    #
    # gps_data.analyse_work_stop_events()
    # gps_data.save_work_stop_events(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/work_stop_events.npy")
    gps_data.load_work_stop_events(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/work_stop_events.npy")

    # gps_data.tag_frame_status()
    # gps_data.save_frame_status(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/work_frame_status.csv")
    gps_data.load_frame_status(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/work_frame_status.csv")


    gps_data.plot_frame_status()

    gps_data.count_by_day()
    gps_data.plot_by_day()
    print("DONE")
    exit(0)

