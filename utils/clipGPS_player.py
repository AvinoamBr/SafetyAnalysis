from datetime import datetime
from matplotlib import pyplot as plt
import cv2
import pynmea2
import pandas as pd

from read_GPS import GPS_Data, datetime_from_file_path, nmea_2latlon, convert_from_ms
import numpy as np

def read_GPS_file(fn):
    '''
    read a single ception GPS file
    parse all $GPVTG messages using pynmea2 parser

    :return
     np.array n*2: [[date_time, speed],...]
    '''
    haversin_speed_data = []
    speed_data = []
    date = datetime_from_file_path(fn)
    year, month, day = date.year, date.month, date.day
    position = [0, np.array([None, None])]
    positions = []
    with open(fn, 'r') as f:
        for l in f.readlines():

            # l = f.readline()
            try:
                ms_from_midnight = int(l.split(",")[0])
            except ValueError: # this is the header line
                continue
            _, hours, minutes, seconds, microseconds = convert_from_ms(ms_from_midnight)
            date_time = datetime(year=year, month=month, day=day, hour=hours, minute=minutes,
                                 second=seconds, microsecond=microseconds)
            ll = ",".join(l.split(",")[1:])
            try:
                parsed = pynmea2.parse(ll)
            except:
                continue
            if ll.startswith("$GPVTG"):  # and parsed.spd_over_grnd_kmph>0:
                speed_value = parsed.spd_over_grnd_kmph
                if speed_value == None:
                    speed_data.append((ms_from_midnight, date_time, None))
                    continue
                speed = float(speed_value)
                speed_data.append((ms_from_midnight, date_time, speed))
            elif ll.startswith("$GNGNS"):
                parsed = pynmea2.parse(ll)
                lat = parsed.lat
                if lat == '':
                    continue
                nmea_lat = parsed.lat
                nmea_lon = parsed.lon
                lat, lon = nmea_2latlon(nmea_lat, nmea_lon)
                positions.append((ms_from_midnight, date_time, lat,lon))
    speed_array = np.array(speed_data)
    positions = np.array(positions)
    positions = pd.DataFrame(positions, columns=['ms_from_midnight','timestamp','lat','lon'])
    speed =  pd.DataFrame(speed_data  , columns=['ms_from_midnight','timestamp','speed'])
    return speed, positions

class GpsVideoPlayer(object):
    FPS = 30
    def __init__(self):
        self.video_file = ""
        self.GPS_file = ""
        pass

    def set_inputs(self, root_directory, time_stamp):
        pass

    def play(self):
        cap = cv2.VideoCapture(self.video_file)
        speed, positions = read_GPS_file(self.GPS_file)
        ms_from_md = speed.iloc[0].ms_from_midnight
        i=0
        while True:
            ret, im = cap.read()
            if ret:
                im = cv2.putText(im, f"{int(ms_from_md)}", (50, 50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)



                # cv2.imshow("im",im)
                # cv2.waitKey(1)
                if i%30==0:
                    speed_index = np.argmin(np.absolute(speed.ms_from_midnight - ms_from_md))
                    speed_from_file = speed.iloc[speed_index].speed

                    pos_index = np.argmin(np.absolute(positions.ms_from_midnight - ms_from_md))
                    lat = positions.iloc[pos_index]['lat']
                    lon = positions.iloc[pos_index]['lon']
                    im = cv2.putText(im, f"speed: {speed_from_file} [KPH]", (250, 50),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)
                    im = cv2.putText(im, f"position: lat:{lat:.8f} lon:{lon:.8f}", (480, 50),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)
                    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                    plt.subplot(1,2,1)
                    plt.imshow(im)
                    plt.draw()
                    plt.subplot(1,2,2)
                    plt.plot(positions.lat,positions.lon)
                    plt.scatter(positions.iloc[pos_index]['lat'], positions.iloc[pos_index]['lon'], s=50, c='r')
                    plt.pause(0.000001)
                    plt.clf()
                ms_from_md += 1/self.FPS * 1000
            else:
                cv2.waitKey(0)
                break
            i+=1
        pass

if __name__ == "__main__":
    gps_video_player = GpsVideoPlayer()
    root = '/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/'
    # gps_video_player.video_file = '/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/videoRootDirectory/20220705/20220705_15/1_20220705-150500_0030h.avi'
    # gps_video_player.GPS_file = '/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/GPS/20220705/NMEARecorder_20220705-150500.csv'

    # time_stamp = "20220616-173500" # a very good example

    time_stamp = "20220616-173000" # a very bad example
    #
    # time_stamp = "20220616-175000" # good example


    # gps_video_player.video_file = f'/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/videoRootDirectory/20220616/20220616_17/1_{time_stamp}_0030h.avi'
    # gps_video_player.GPS_file = f'/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/GPS/20220616/NMEARecorder_{time_stamp}.csv'

    # mesubim good example:
    root ='/media/performance/Data/Cemex/Readymix/Mesubim/Shuff_4_CAT_926M_20220606-20220616'
    time_stamp = "20220606-150000"

    # mesubim bad example:
    root = '/media/performance/Data/Cemex/Readymix/Mesubim/Shuff_4_CAT_926M_20220707-20220725/'
    time_stamp = "20220707-161500"



    gps_video_player.video_file = f'{root}/videoRootDirectory/{time_stamp[:8]}/{time_stamp[:8]}_{time_stamp[9:11]}/1_{time_stamp}_0030h.avi'
    gps_video_player.GPS_file = f'{root}/GPS/{time_stamp.split("-")[0]}/NMEARecorder_{time_stamp}.csv'

    print(gps_video_player.video_file)
    print(gps_video_player.GPS_file)


    gps_video_player.play()

