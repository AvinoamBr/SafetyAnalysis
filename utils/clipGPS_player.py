import glob
import logging
import os.path
import time
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import cv2
import pynmea2
import pandas as pd
import matplotlib
import matplotlib.animation as animation

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
    with open(fn, 'r', errors='ignore') as f:
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
                if speed_value == None or speed_value=='\n':
                    speed_data.append((ms_from_midnight, date_time, None))
                    continue
                speed = float(speed_value)
                speed_data.append((ms_from_midnight, date_time, speed))
            elif ll.startswith("$GPGGA"):
                parsed = pynmea2.parse(ll)
                lat = parsed.lat
                if lat == '':
                    continue
                nmea_lat = parsed.lat
                nmea_lon = parsed.lon
                try:
                    lat, lon = nmea_2latlon(nmea_lat, nmea_lon)
                except:
                    logging.warning(f"failed to parse GPS line: {ll  }")
                    continue
                positions.append((ms_from_midnight, date_time, lat,lon))
    speed_array = np.array(speed_data)
    positions = np.array(positions)
    positions = pd.DataFrame(positions, columns=['ms_from_midnight','timestamp','lat','lon'])
    speed =  pd.DataFrame(speed_data  , columns=['ms_from_midnight','timestamp','speed'])
    return speed, positions

class GpsVideoPlayer(object):
    fps = 30
    output_fps=30
    def __init__(self):
        self._clip_name = None
        self._video_sync_file = ""
        self._video_file = ""
        self._GPS_file = ""
        self.output = None

    @property
    def video_file(self, video_file):
        return self._video_file

    @video_file.setter
    def video_file(self, video_file):
        self._video_file = video_file
        self._video_sync_file = video_file.replace(".avi",".csv")
        self._clip_name = os.path.split(video_file)[1].split("_")[1]
        self._out_file = video_file.replace(os.path.split(video_file)[0],f"{self.output}/")
        pass

    @property
    def GPS_file(self):
        return self._GPS_file
    @GPS_file.setter
    def GPS_file(self, GPS_file):
        self._GPS_file = GPS_file

    def set_inputs(self, root_directory, time_stamp):
        pass

    def play(self):
        cap = cv2.VideoCapture(self._video_file)
        try:
            speed, positions = read_GPS_file(self._GPS_file)
        except:
            logging.warning(f"failed to get GPS data from file {self._GPS_file}")
            return
        # ms_from_md = speed.iloc[0].ms_from_midnight
        ms_from_md = pd.read_csv(self._video_sync_file).iloc[0].millisecondsFromMidnight
        i=0
        if self.output:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # float `width`
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output = cv2.VideoWriter(self._out_file, fourcc, self.output_fps, 1, (width,height))

        while True:
            ret, im = cap.read()
            # fig  = plt.figure()
            if ret:
                if i%self.output_fps==0:
                    speed_index = np.argmin(np.absolute(speed.ms_from_midnight - ms_from_md))
                    speed_from_file = speed.iloc[speed_index].speed

                    pos_index = np.argmin(np.absolute(positions.ms_from_midnight - ms_from_md))
                    ms_GPS =  positions.ms_from_midnight[pos_index]
                    ms_diff = int(ms_from_md - ms_GPS)
                    lat = positions.iloc[pos_index]['lat']
                    lon = positions.iloc[pos_index]['lon']
                    im = cv2.putText(im, f"ms_clip {int(ms_from_md)}", (50, 50),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)
                    im = cv2.putText(im, f"ms_GPS: {ms_GPS}", (50, 75),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 100, 100), 2, cv2.LINE_AA)
                    im = cv2.putText(im, f"ms_diff: {ms_diff}", (50, 100),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 100, 100), 2, cv2.LINE_AA)
                    im = cv2.putText(im, f"frame: {i} clip_time: {int(divmod(i/30,60)[0]):02d}:{int(divmod(i/30,60)[1]):02d}", (50, 150),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 100, 100), 2, cv2.LINE_AA)
                    im = cv2.putText(im, f"speed: {speed_from_file} [KPH]", (250, 50),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)
                    im = cv2.putText(im, f"position: lat:{lat:.8f} lon:{lon:.8f}", (480, 50),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2, cv2.LINE_AA)

                    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
                    plt.subplot(1,2,1)
                    plt.suptitle(self._clip_name)
                    plt.imshow(im)
                    plt.draw()
                    ax = plt.subplot(1,2,2)
                    plt.plot(positions.lon,positions.lat)
                    plt.scatter(positions.iloc[pos_index]['lon'], positions.iloc[pos_index]['lat'], s=50, c='r')
                    plt.pause(0.0000000001)
                    ax.set_aspect('equal')
                    # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.clf()
                    # output.write(data)
                    # plt.close()
                ms_from_md += 1 / self.fps * 1000
            else:
                break
            i+=1
        time.sleep(10)
        pass

if __name__ == "__main__":
    gps_video_player = GpsVideoPlayer()
    time_stamp = None
    date = None
    # gps_video_player.video_file = '/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/videoRootDirectory/20220705/20220705_15/1_20220705-150500_0030h.avi'
    # gps_video_player.GPS_file = '/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/GPS/20220705/NMEARecorder_20220705-150500.csv'

    # time_stamp = "20220616-173500" # a very good example
    if False:
        root = '/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/'
        time_stamp = "20220616-173000" # a very bad example
    #
    # time_stamp = "20220616-175000" # good example


    # gps_video_player.video_file = f'/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/videoRootDirectory/20220616/20220616_17/1_{time_stamp}_0030h.avi'
    # gps_video_player.GPS_file = f'/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220616-20220707/GPS/20220616/NMEARecorder_{time_stamp}.csv'

    # mesubim good example:
    if False:
        root ='/media/performance/Data/Cemex/Readymix/Mesubim/Shuff_4_CAT_926M_20220606-20220616'
        time_stamp = "20220606-150000"

    # mesubim bad example:
    if True:
        root = '/media/performance/Data/Cemex/Readymix/Mesubim/Shuff_4_CAT_926M_20220707-20220725/'
        # time_stamp = "20220707-161500"
        # date = "20220707" # bad
        date = "20220708"



    #Modiim good
    if False:
        root = '/media/performance/Data/Cemex/Readymix/Modiim/Shuff_4_CAT_972M_20220616-20220703/'
        time_stamp = '20220616-144000'

    #Modiim good
    if False:
        root = '/media/performance/Data/Cemex/Readymix/Modiim/Shuff_4_CAT_972M_20220616-20220703/'
        time_stamp = '20220623-051000'


    #Modiim good
    if False:
        root = '/media/performance/Data/Cemex/Readymix/Modiim/Shuff_4_CAT_972M_20220616-20220703/'
        time_stamp = '20220623-045500'


    #Modiim good
    if False:
        root = '/media/performance/Data/Cemex/Readymix/Modiim/Shuff_4_CAT_972M_20220616-20220703/'
        time_stamp = '20220623-173000'

    if time_stamp:
        gps_video_player._video_file = f'{root}/videoRootDirectory/{time_stamp[:8]}/{time_stamp[:8]}_{time_stamp[9:11]}/1_{time_stamp}_0030h.avi'
        gps_video_player._GPS_file = f'{root}/GPS/{time_stamp.split("-")[0]}/NMEARecorder_{time_stamp}.csv'

        print(gps_video_player._video_file)
        print(gps_video_player._video_file)
        print(gps_video_player._GPS_file)


        gps_video_player.play()

    elif date:
        videos = glob.glob(f'{root}/videoRootDirectory/{date}/*/1_*.avi')
        for video in sorted(videos):
            logging.info(f"video file: {video}")
            time_stamp = os.path.split(video)[1][2:17]
            time_stamp_obj = datetime.strptime(time_stamp, "%Y%m%d-%H%M%S")
            GPS_to_clip_offset = np.arange(-10,10)
            for offset in GPS_to_clip_offset:
                time_stamp_obj_with_offset = time_stamp_obj + timedelta(seconds=int(offset))
                time_stamp_str_with_offset = datetime.strftime(time_stamp_obj_with_offset,  "%Y%m%d-%H%M%S")
                GPS_file = f'{root}/GPS/{date}/NMEARecorder_{time_stamp_str_with_offset}.csv'
                if os.path.exists(GPS_file):
                    break
            else:
                logging.warning(f"missing GPS file {GPS_file}")
                continue
            # gps_video_player.output = '/home/ception/workspace/GPS_validation/Mesubim/'
            gps_video_player.video_file = video
            gps_video_player.GPS_file = GPS_file
            gps_video_player.play()