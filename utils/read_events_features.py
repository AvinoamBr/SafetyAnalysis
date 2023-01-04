import logging

import pandas as pd
import numpy as np
import datetime
import os
from utils.clipGPS_player import read_GPS_file
from matplotlib import pyplot as plt

time_str_format = "%Y%m%d-%H%M%S"
date_format = "%Y%m%d"
records = ["Shuff_4_CAT_972M_20220518-20220606",
           "Shuff_4_CAT_972M_20220606-20220616",
           "Shuff_4_CAT_972M_20220616-20220703",
           "Shuff_4_CAT_972M_20220704-20220707",
           "Shuff_4_CAT_972M_20220707-20220710",
           "Shuff_4_CAT_972M_20220711-20220720"]
root_records = [f'/media/performance/Data/Cemex/Readymix/Modiim/{rec}' for rec in records]
FPS = 30


def read_events_GPS(e):
    day = int(e.day)
    month = int(e.month)
    year = int(e.year)
    timeOfEvent = datetime.datetime.strptime(e['Time of event '], '%H:%M:%S').time()
    hour, minute, second = timeOfEvent.hour, timeOfEvent.minute, timeOfEvent.second
    minute_round = divmod(minute,5)[0]*5
    date_str = f"{year:04d}{month:02d}{day:02d}"
    datetime_str = datetime.datetime.strftime(datetime.datetime(year,month,day,hour,minute_round,0),  '%Y%m%d-%H%M%S')
    # get GPS_data_path
    for root_record in root_records:
        GPS_data_path = f"{root_record}/GPS/{date_str}/NMEARecorder_{datetime_str}.csv"
        if os.path.exists(GPS_data_path):
            logging.info(f"GPS file: {GPS_data_path}")
            break
    else:
        logging.warning(f"failed to find GPS data for event {e}")

    # get video_frameTo_ms_from_md
    video_frameTo_ms_from_md_file = f"{root_record}/videoRootDirectory/{date_str}/{date_str}_{hour:02d}/1_{datetime_str}_0030h.csv"
    if not os.path.exists(video_frameTo_ms_from_md_file):
        logging.warning(f'missing video-to-msFromMidnight file in {video_frameTo_ms_from_md_file}')
    video_frameTo_ms_from_md = pd.read_csv(video_frameTo_ms_from_md_file)
    # /media/performance/Data/Cemex/Readymix/Modiim/Shuff_4_CAT_972M_20220711-20220720/videoRootDirectory/20220711/20220711_13/1_20220711-132000_0030h.csv
    index = ((minute-minute_round)*60+second) * FPS
    ms_from_md =video_frameTo_ms_from_md.iloc[index].millisecondsFromMidnight

    speed,location = read_GPS_file(GPS_data_path)

    speed_index = np.argmin(np.absolute(speed.ms_from_midnight - ms_from_md))
    speed_from_file = speed.iloc[speed_index].speed
    pos_index = np.argmin(np.absolute(location.ms_from_midnight - ms_from_md))
    lat = location.iloc[pos_index]['lat']
    lon = location.iloc[pos_index]['lon']
    event = {'speed':speed_from_file, 'lat':lat, 'lon':lon, 'hour':hour, 'severity':e.Severity, 'Type of Event': e['Type of Event']}
    return event

def heuristic_events_counter(events):
    # 2d histogram over position
    positions = events[['lon','lat']].values
    lon, lat = positions[:,0], positions[:,1]

    position_hist = np.histogram2d(lon,lat)
    H, lon_edges, lat_edges = position_hist

    return H, lon_edges, lat_edges


if __name__ == "__main__":
    '''
    events_file = '/media/backup/Algo/users/avinoam/Cemex_manager/ModiimEvents.csv'
    events_list = pd.read_csv(events_file)
    events = []
    for (i,e) in events_list.iterrows():
        event = read_events_GPS(e)
        events.append(event)
    '''

    # events_df = pd.DataFrame(events)
    events_df = pd.read_csv('/media/backup/Algo/users/avinoam/Cemex_manager/ModiimEvents_DataFrame.csv')
    map_fn = '/home/ception/Downloads/Govmap/Govmap.png'
    map = plt.imread(map_fn)
    ax = plt.subplot(1,2,1)
    plt.imshow(map, extent = ( 34.966404, 34.980377, 32.021716,32.012796 ), aspect='equal')

    H, lon_edges, lat_edges = heuristic_events_counter(events_df)
    X, Y = np.meshgrid(lon_edges, lat_edges)
    ax.pcolormesh(X, Y, H.T, alpha=0.5, cmap='Reds')
    # plt.subplot(1,2,2)
    # plt.imshow(H.T, extent=(lon_edges[0],lon_edges[-1],lat_edges[-1],lat_edges[0]), alpha=0.2, origin = 'upper',
    #            aspect='equal')
    plt.scatter(events_df.lon, events_df.lat, alpha=0.8)
    plt.show()
    print()

