import glob
import logging

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import datetime
import os
from utils.clipGPS_player import read_GPS_file
from matplotlib import pyplot as plt
from matplotlib import colors


time_str_format = "%Y%m%d-%H%M%S"
date_format = "%Y%m%d"
RECORDS ={'modiim': ["Shuff_4_CAT_972M_20220518-20220606",
           "Shuff_4_CAT_972M_20220606-20220616",
           "Shuff_4_CAT_972M_20220616-20220703",
           "Shuff_4_CAT_972M_20220704-20220707",
           "Shuff_4_CAT_972M_20220707-20220710",
           "Shuff_4_CAT_972M_20220711-20220720"]}
root_records = [f'/media/performance/Data/Cemex/Readymix/Modiim/{rec}' for rec in RECORDS['modiim']]
FPS = 30

class EventDataMining(object):
    '''
    read event raw data containing basic info
    extract additional data from recorder data
    '''

    # path pointer to look for data
    # dict { site_name<str>: [  root_dir_1, root_dir_2 ....],...}
    RECORDS = RECORDS

    def __init__(self):
        self._event_dict = {}
        self._raw_event = None
        self._video_frameTo_ms_from_md_file = [] # list of paths for video sync files
        self._video_files = [] # list of paths for video files
        self._GPS_data_path = None
        self._root_record_path = None
        self._datetime_of_event = None # place holder for datetime object <datetime.datetime>

    @property
    def event_dict(self):
        return self._event_dict

    def read_events_datetime(self, e):
        self._raw_event  = e
        # extract all datetime variables
        day = int(e.day)
        month = int(e.month)
        year = int(e.year)
        timeOfEvent = datetime.datetime.strptime(e['Time of event '], '%H:%M:%S').time()
        hour, minute, second = timeOfEvent.hour, timeOfEvent.minute, timeOfEvent.second
        self._datetime_of_event = datetime.datetime(year,month,day,hour,minute,second)

    def _datetime_of_event_extract(self):
        doe = self._datetime_of_event
        day, month, year, hour, minute, second = (
            doe.day,
            doe.month,
            doe.year,
            doe.hour,
            doe.minute,
            doe.second)
        return (day, month, year, hour, minute, second)


    def set_recorder_paths(self):
        RECORDER_DATETIME_FORMAT = '%Y%m%d-%H%M%S'
        RECORDER_DATE_FORMAT = '%Y%m%d'

        day, month, year, hour, minute, second  = self._datetime_of_event_extract()
        minute_round = divmod(minute,5)[0]*5
        file_datetime = datetime.datetime(year,month,day,hour,minute_round,0)

        recorder_date_str = datetime.datetime.strftime(file_datetime, RECORDER_DATE_FORMAT ) # YYYYMMDD format for path
        recorder_datetime_str = datetime.datetime.strftime(file_datetime, RECORDER_DATETIME_FORMAT )

        for root_record in root_records:
            GPS_data_path = f"{root_record}/GPS/{recorder_date_str}/NMEARecorder_{recorder_datetime_str}.csv"
            if os.path.exists(GPS_data_path):
                logging.debug(f"GPS file: {GPS_data_path}")
                self._root_record_path = root_record
                self._GPS_data_path = GPS_data_path
                video_path =  f"{root_record}/videoRootDirectory/{recorder_date_str}/{recorder_date_str}_{hour:02d}/"
                self._video_files = glob.glob(f"{video_path}/*{recorder_datetime_str}*.avi")
                self._video_frameTo_ms_from_md_file = glob.glob(f"{video_path}/*{recorder_datetime_str}*.csv")
                break
        else:
            logging.warning(f"failed to find GPS data for event {e}")
            pass
        if self._video_files == []:
            logging.warning(f"failed to find video files")

    @property
    def seconds_from_video_start(self):
        (day, month, year, hour, minute, second) = self._datetime_of_event_extract()
        minute_round = divmod(minute,5)[0]*5
        return (minute-minute_round)*60+second

    @property
    def vlc_command(self, index=1):
        return f"vlc {self._video_files[index]} --start-time {self.seconds_from_video_start}"

    def add_data_from_GPS(self):
        video_frameTo_ms_from_md = pd.read_csv(self._video_frameTo_ms_from_md_file[0])

        second_from_video_start = self.seconds_from_video_start
        index = second_from_video_start * FPS
        ms_from_md =video_frameTo_ms_from_md.iloc[index].millisecondsFromMidnight

        speed,location = read_GPS_file(self._GPS_data_path)

        speed_index = np.argmin(np.absolute(speed.ms_from_midnight - ms_from_md))
        speed_from_file = speed.iloc[speed_index].speed
        pos_index = np.argmin(np.absolute(location.ms_from_midnight - ms_from_md))
        lat = location.iloc[pos_index]['lat']
        lon = location.iloc[pos_index]['lon']
        (_, _, _, hour, _, _) = self._datetime_of_event_extract()
        self._event_dict = {'speed':speed_from_file, 'lat':lat, 'lon':lon, 'hour':hour, 'time':self._datetime_of_event,
                 'severity':e.Severity, 'Type of Event': e['Type of Event'], 'vlc_command':self.vlc_command}


class EventDataMining_new(EventDataMining):
    def read_events_datetime(self, e):
        self._raw_event  = e
        # extract all datetime variables
        day = int(e['Clip day'])
        month = int(e['Clip month'])
        year = int(e['Clip year'])
        hour = int(e['Clip hour'])
        minute = int(e['Clip minute'])
        second =int(e['Clip sec'])

        self._datetime_of_event = datetime.datetime(year,month,day,hour,minute,second)

    def add_data_from_GPS(self):
        video_frameTo_ms_from_md = pd.read_csv(self._video_frameTo_ms_from_md_file[0])

        second_from_video_start = self.seconds_from_video_start
        index = second_from_video_start * FPS
        ms_from_md =video_frameTo_ms_from_md.iloc[index].millisecondsFromMidnight

        speed,location = read_GPS_file(self._GPS_data_path)

        speed_index = np.argmin(np.absolute(speed.ms_from_midnight - ms_from_md))
        speed_from_file = speed.iloc[speed_index].speed
        pos_index = np.argmin(np.absolute(location.ms_from_midnight - ms_from_md))
        lat = location.iloc[pos_index]['lat']
        lon = location.iloc[pos_index]['lon']
        (_, _, _, hour, _, _) = self._datetime_of_event_extract()
        e = self._raw_event
        self._event_dict = {'speed':speed_from_file, 'lat':lat, 'lon':lon, 'hour':hour, 'time':self._datetime_of_event,
                 'severity':e.Severity, 'Type of Event': e['obstacles_types'], 'vlc_command':self.vlc_command}

    def set_recorder_paths(self):
        RECORDER_DATETIME_FORMAT = '%Y%m%d-%H%M%S'
        RECORDER_DATE_FORMAT = '%Y%m%d'
        #
        day, month, year, hour, minute, second  = self._datetime_of_event_extract()
        minute_round = divmod(minute,5)[0]*5
        file_datetime = datetime.datetime(year,month,day,hour,minute_round,0)
        #
        recorder_date_str = datetime.datetime.strftime(file_datetime, RECORDER_DATE_FORMAT ) # YYYYMMDD format for path
        recorder_datetime_str = datetime.datetime.strftime(file_datetime, RECORDER_DATETIME_FORMAT )
        root_record = self._raw_event.Recordings_folder.strip()
        GPS_data_path = f"{root_record}/GPS/{recorder_date_str}/NMEARecorder_{recorder_datetime_str}.csv"
        logging.debug(f"GPS file: {GPS_data_path}")
        self._root_record_path = root_record
        self._GPS_data_path = GPS_data_path
        video_path =  f"{root_record}/videoRootDirectory/{recorder_date_str}/{recorder_date_str}_{hour:02d}/"
        self._video_files = glob.glob(f"{video_path}/*{recorder_datetime_str}*.avi")
        self._video_frameTo_ms_from_md_file = glob.glob(f"{video_path}/*{recorder_datetime_str}*.csv")
        if len(self._video_files) <=1 :
            logging.warning(f"found {len(self._video_files)} video clips for event")


if __name__ == "__main__":
    EXTRACT_EVENTS = False
    if EXTRACT_EVENTS:
        events_file = '/media/backup/Algo/users/avinoam/Cemex_manager/ModiimEvents.csv'
        events_list = pd.read_csv(events_file)
        event_dicts = []
        event_data_mining = EventDataMining()
        for (i,e) in events_list.iterrows():
            event_data_mining.read_events_datetime(e)
            event_data_mining.set_recorder_paths()
            event_data_mining.add_data_from_GPS()
            event_dicts.append(event_data_mining.event_dict)

        events_df = pd.DataFrame(event_dicts)
        events_df.to_csv('/media/backup/Algo/users/avinoam/Cemex_manager/ModiimEvents_DataFrame.csv')
    else:
        events_df = pd.read_csv('/media/backup/Algo/users/avinoam/Cemex_manager/ModiimEvents_DataFrame.csv')

    EXTRACT_EVENTS_NEW = True
    if EXTRACT_EVENTS_NEW:
        events_file = '/home/ception/Downloads/csv_file_2023_01_08_10_03_20.csv'
        events_list = pd.read_csv(events_file)
        events_list = events_list[events_list[' MSG ID']==11]
        event_dicts = []
        event_data_mining = EventDataMining_new()
        N_events = len(events_list)
        for (i,(row,e)) in enumerate(events_list.iterrows()):
            logging.info(f"event {i}/{N_events}")
            event_data_mining.read_events_datetime(e)
            event_data_mining.set_recorder_paths()
            event_data_mining.add_data_from_GPS()
            event_dicts.append(event_data_mining.event_dict)
        events_df = pd.DataFrame(event_dicts)

    events_df = events_df[:-1]
    map_fn = '/home/ception/Downloads/Govmap/Govmap.png'
    map = plt.imread(map_fn)
    # ax = plt.subplot(2,2,1)
    plt.imshow(map, extent = ( 34.966404, 34.980377, 32.021716,32.012796 ), aspect='equal')

    H, lon_edges, lat_edges = position_hist_2d(events_df)
    X, Y = np.meshgrid(lon_edges, lat_edges)
    bounds = [0,1,2,3,5,10,20,30]
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    hist2d = H.T
    hist2d = np.ma.masked_where(hist2d < 1, hist2d)
    plt.imshow(hist2d, extent=(lon_edges[0], lon_edges[-1], lat_edges[-1], lat_edges[0]), alpha=0.5, origin ='upper',
               aspect='equal', norm=norm, cmap='Reds')#plt.cm.get_cmap('Reds', H.max()+1))
    plt.colorbar()
    # for e_type in ['Light Vehicle', 'Heavy Vehicle', 'Person']:
    #     df = events_df[events_df['Type of Event']==e_type]
    #     plt.scatter(df.lon, df.lat, alpha=0.8, label=e_type)
    df = events_df
    plt.scatter(df.lon, df.lat, alpha=0.8)
    plt.xlabel('Lon [deg]')
    plt.ylabel('Lat [deg]')
    plt.title('Geo-Grid density of events')
    plt.show()

    # plt.subplot(2,2,2)
    bins = np.arange(24,step=2)
    bottom = np.zeros(bins.shape[0]-1)
    for e_type in events_df['Type of Event'].value_counts().index:
        df = events_df[events_df['Type of Event']==e_type]
        h, *_ = plt.hist(df.hour, bins=bins, bottom=bottom, label=e_type)
        bottom += h
        # df.hour.hist(bins=bins, bottom=bottom)
    plt.legend()
    plt.title('Events historgram in day-hour')
    plt.xlabel("time in day [hr]")
    plt.ylabel("N events")
    plt.show()
    # plt.subplot(2,2,3)
    plt.imshow(map, extent = ( 34.966404, 34.980377, 32.021716,32.012796 ), aspect='equal')
    plt.xlabel('Lon [deg]')
    plt.ylabel('Lat [deg]')
    plt.title('DB-scan clustering of events')

    # events_df = events_df[events_df.severity==3]
    labels = position_clustering_dbscan(events_df)
    colors = np.random.randint(0,225,(max(labels)+1,3))/255
    df = events_df[labels!=-1]
    c = colors[labels[labels!=-1]]
    plt.scatter(df.lon, df.lat, alpha=0.8, c = labels[labels!=-1], cmap=plt.cm.get_cmap('cubehelix', labels.max()+1))
    df = events_df[labels==-1]
    c = np.array((20,100,100))/255
    plt.scatter(df.lon, df.lat, alpha=0.8, c=c, marker='*', s=15)
    c = np.array((200,100,100))/255
    plt.scatter(df.lon, df.lat, alpha=0.8, c=c, marker='*', s=5)
    plt.show()



