import glob
import logging
import os
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pynmea2
from sklearn.metrics.pairwise import haversine_distances
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

def nmea_2latlon(nmea_lat,nmea_lon):
    def _nmea_2_deg(nmea):
        nmea = float(nmea)
        deg = int(nmea/100)
        sec = nmea - deg * 100
        return deg + sec/60
    return (_nmea_2_deg(nmea_lat), _nmea_2_deg(nmea_lon))

class GPS_Data(object):
    def __init__(self, site, iopath, *args, **kwargs):
        self.frame_work_status = pd.DataFrame()
        self.speed_array = np.array([]) # place holder for : np.array n*2: [[date_time, speed],...]
        self.site = site
        self.iopath = iopath

    def _read_GPS_file(self, fn):
        '''
        read a single ception GPS file
        parse all $GPVTG messages using pynmea2 parser

        :return
         np.array n*2: [[date_time, speed],...]
        '''
        haversin_speed_data = []
        gps_data = []
        date = datetime_from_file_path(fn)
        year,month,day = date.year,date.month,date.day
        position = [0, np.array([None,None])]
        positions = []
        with open(fn,'r', errors='ignore') as f:
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
                    elif ll.startswith("$GNGNS"):
                        parsed = pynmea2.parse(ll)
                        lat = parsed.lat
                        if lat=='':
                            continue
                        nmea_lat = parsed.lat
                        nmea_lon = parsed.lon
                        lat,lon = nmea_2latlon(nmea_lat,nmea_lon)
                        positions.append((lat,lon))
                        pos_rad = np.deg2rad([lat,lon])
                        new_postion = (ms_from_midnight,pos_rad)
                        if position[1].any():
                            EARTH_RADIUS = 6371* 1000 # [m]
                            MS_TO_SEC = 1000
                            MPS_TO_KPH = 3.6
                            UNKNOWN_BUG_FACTOR = 1.0
                            # speed_dx = haversine_distances(np.array([new_postion[1],position[1]]))[0][1]\
                            #            * EARTH_RADIUS /  (new_postion[0]-position[0]) * MS_TO_SEC * MPS_TO_KPH *UNKNOWN_BUG_FACTOR
                            # print(f"{speed:.2f}, {speed_dx:.2f} mode_indicator:{parsed.mode_indicator}")
                            # haversin_speed_data.append((date_time,speed_dx))
                        position = new_postion

                except Exception as e:
                    continue
        speed_array = np.array(gps_data)
        haversin_speed_array = np.array(haversin_speed_data)
        # plt.plot(speed_array.T[0],speed_array.T[1], label='speed GPVTG')
        # plt.plot(haversin_speed_array.T[0],haversin_speed_array.T[1], label='speed GNGNS position diff')
        # plt.xlabel("time")
        # plt.ylabel("KPH")
        # plt.title(os.path.split(fn)[1])
        # plt.legend()
        # plt.show()

        positions = np.array(positions).T
        # plt.plot(positions[0],positions[1])
        # plt.show()

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
                print(e)
                continue
            if speeds.shape[0] != 0:
                speed_array = np.row_stack((speed_array, speeds))

        self.speed_array = speed_array
        self.gps_df = pd.DataFrame(self.speed_array, columns=['datetime', 'speed'])

    def save(self, fn=None):
        fn = fn or f"{self.iopath}/speed.npy"
        np.save(fn, self.speed_array)

    def load(self, fn=None):
        fn = fn or f"{self.iopath}/speed.npy"
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
        numevents = self.events.shape[0]
        for i,e in self.events.iterrows():
            continous = is_event_continous(e,  self.gps_df) # event is considered continous if signal is not stopping for more that one minute.
            print(f"{i}/{numevents} duration: {e.duration}, continous: {continous}")
            continous_events.append(continous)
        self.events['continous_events'] = continous_events

        return self.gps_df, self.events

    def save_work_stop_events(self, fn=None):
        fn = fn or f"{self.iopath}/work_stop_events.npy"
        self.events.to_csv(fn)

    def load_work_stop_events(self, fn=None):
        fn = fn or f"{self.iopath}/work_stop_events.npy"
        self.events = pd.read_csv(fn)

    def tag_frame_status(self):
        '''
        using the stop events, tag ANY_FRAME if is
            working (velocity>0 or short stops)
            pause (stops with engine on)
            off (stops, before or after engine power-off

        :return:
        '''

        self.frame_work_status = pd.Series(["worked"] * self.gps_df.shape[0])
        for (i, e) in self.events.iterrows():
            if not e.continous_events:
                self.frame_work_status[e.stopped_ids:e.started_ids] = 'off'
            elif e.pause_event:
                self.frame_work_status[e.stopped_ids:e.started_ids] = 'pause'
        self.frame_work_status[self.gps_df.speed.isna()] = 'missing_NA'

    def save_frame_status(self, fn=None):
        fn = fn or f"{self.iopath}/work_frame_status.csv"
        self.frame_work_status.to_csv(fn, index=False,header=False)

    def load_frame_status(self, fn=None):
        fn = fn or f"{self.iopath}/work_frame_status.csv"
        self.frame_work_status = pd.read_csv(fn, names=['frame_work_status'],dtype=str).frame_work_status

    def plot_frame_status(self, save=False):
        gps_df = self.gps_df
        for (work_status,color) in [('off','r'), ('worked','g'), ('pause','y')]:
            speed = (gps_df[self.frame_work_status == work_status]).speed
            d = gps_df[self.frame_work_status == work_status].datetime
            plt.scatter(d, speed, label=work_status, s=1, color=color)
        d = gps_df[self.frame_work_status == 'missing_NA'].datetime
        plt.scatter(d, np.zeros(d.shape[0]), label=work_status, s=1, color='k')
        plt.legend()
        plt.title(self.site)
        if save:
            fn = f"{self.iopath}/frame_status.png"
            plt.savefig(fn)
        plt.show()

    def count_by_day(self):
        worked = self.frame_work_status
        self.gps_df['date'] = self.gps_df['datetime'].apply(lambda dt: dt.date())
        worked_by_day = self.gps_df[worked == 'worked']['date'].value_counts().sort_index() / (60*60)
        pause_by_day = self.gps_df[worked == 'pause']['date'].value_counts().sort_index() / (60*60)
        off_by_day = self.gps_df[worked == 'off']['date'].value_counts().sort_index() / (60*60)
        missing_by_day = self.gps_df[worked == 'missing_NA']['date'].value_counts().sort_index() / (60*60)

        start_date = min((worked_by_day.index[0], pause_by_day.index[0], off_by_day.index[0]))
        end_date = max((worked_by_day.index[-1], pause_by_day.index[-1], off_by_day.index[-1]))

        all_dates = np.arange(start_date, end_date)
        all_dates = np.append(all_dates, end_date)

        # add zeros were there is missing data, to enable bars bottom
        for work_status in [worked_by_day, pause_by_day, off_by_day, missing_by_day]:
            for d in all_dates:
                if d not in work_status.index.values:
                    work_status.loc[d] = 0

        self.pause_by_day = pause_by_day.sort_index()
        self.worked_by_day = worked_by_day.sort_index()
        self.off_by_day = off_by_day.sort_index()
        self.missing_by_day = missing_by_day.sort_index()


    def plot_by_day(self, save=False):
        plt.figure(figsize=(25,15))
        plt.bar(x=self.worked_by_day.index, height=self.worked_by_day, label='work',
                color='g')
        plt.bar(x=self.pause_by_day.index, height=self.pause_by_day, label='pause',
                bottom=self.worked_by_day, color='y')
        plt.bar(x=self.off_by_day.index, height=self.off_by_day, label='parking',
                bottom=self.pause_by_day + self.worked_by_day, color='r')
        plt.bar(x=self.missing_by_day.index, height=self.missing_by_day, label='missing_NA',
                bottom=self.pause_by_day + self.worked_by_day + self.off_by_day, color='k')
        plt.legend()
        plt.title(self.site)
        ticks = np.arange(self.off_by_day.index[0]-timedelta(self.off_by_day.index[0].weekday()+1) ,self.off_by_day.index[-1], timedelta(days=7))
        plt.xticks(ticks, rotation = 45)
        if save:
            fn = f"{self.iopath}/plot_by_day.png"
            plt.savefig(fn)
        plt.show()



if __name__ == "__main__":
    site = "PetachTikva" # not ready
    # site = "Golani"
    # site = "Modiim"
    # site = "Mesubim" # not ready


    # path = f"/media/performance/Data/Cemex/Readymix/{site}/*/GPS/2022*/**"
    # path = "/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/**"
    path ='/media/performance/Data/Cemex/Readymix/Mesubim/Shuff_4_CAT_926M_20220606-20220616/*'
    # path = "/media/performance/Data/Cemex/Readymix/Mesubim/Shuff_4_CAT_926M_20220707-20220725/**"
    GPS_files_path = f"/media/performance/Data/Cemex/Readymix/Mesubim/Shuff_4_CAT_926M_20220606-20220616/GPS/*/*"

    # path ='/media/performance/Data/Cemex/Readymix/Modiim/Shuff_4_CAT_972M_20220616-20220703/'
    # GPS_files_path = f"{path}/GPS/20220623/*"

    # path ='/media/performance02/Data/Cemex_venture/PetachTikva/Cat_930M_PT/20221228-20221221/'
    # GPS_files_path = f"{path}/GPS/20221216/*"

    path ='/media/performance/Data/Cemex_venture/PetachTikva/Cat_930M_PT/'
    GPS_files_path = f"{path}/GPS/20221123/*"

    iopath = f"/media/backup/Algo/users/avinoam/safety_analysis/{site}"
    os.makedirs(iopath, exist_ok=True)
    gps_data = GPS_Data(site, iopath)

    gps_data.read_GPS_files_from_folder(GPS_files_path)
    # gps_data.save(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/speed.npy")
    # gps_data.load(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/speed.npy")
    #
    gps_data.analyse_work_stop_events()
    # gps_data.save_work_stop_events(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/work_stop_events.npy")
    # gps_data.load_work_stop_events(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/work_stop_events.npy")

    gps_data.tag_frame_status()
    # gps_data.save_frame_status(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/work_frame_status.csv")
    # gps_data.load_frame_status(f"/media/backup/Algo/users/avinoam/safety_analysis/{site}/work_frame_status.csv")

    gps_data.plot_frame_status(save=False)

    gps_data.count_by_day()
    gps_data.plot_by_day(save=False)
    print("DONE")
    exit(0)

