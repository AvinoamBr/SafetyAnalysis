import glob
import logging
from datetime import datetime, timedelta
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pynmea2
from matplotlib import dates

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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


def read_GPS_file(fn):
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

def read_events_table(fn):
    csv = pd.read_csv(fn)
    csv = csv[csv.Siteid=='yavne']
    csv['datetime'] = pd.to_datetime(csv['Creation Time'],format='%Y-%m-%dT%H:%M:%S')
    csv['hour'] = csv['datetime'].apply(lambda d:d.hour + d.minute/60)
    return csv

events_fn = "/media/backup/Algo/users/avinoam/Cemex_manager/query_result_2022-11-03T09_42_47.268887Z.csv"
events = read_events_table(events_fn)

speed_csv_path = "/media/backup/Algo/users/avinoam/safety_analysis/yavne_speed.csv"
# path  = "/media/performance/Data/Cemex/Readymix/Yavne/DL250_20220526-20220607/**"
# path = "/media/performance/Data/Cemex/Readymix/Yavne/DL250_*/GPS/*/**"
path = "/media/performance/Data/Cemex/Readymix/Modiim/*/GPS/*/**"

GPS_files = glob.glob(f"{path}/NMEARecorder_20220*", recursive=True)
GPS_files = sorted(GPS_files)

speed_array = np.array([[None,None]])
for fn in sorted(GPS_files):
    datetime_from_file_path(fn)
    logging.info(f"read file {fn}")
    try:
        speeds = read_GPS_file(fn)
    except Exception as e:
        continue
    if speeds.shape[0]!=0:
        speed_array = np.row_stack((speed_array,speeds))


def gps_array_to_stop_events(gps_df):
    def is_event_continous(event, gps_df):
        events_gps = gps_df[np.logical_and(gps_df.datetime >= event.ends, gps_df.datetime <= event.starts)]
        if events_gps.shape[0]==1:
            return True
        if not (event.pause_event):
            return True
        return np.max(events_gps.datetime.diff()) <= timedelta(minutes=1)

    gps_array = gps_df.values
    gps_df = pd.DataFrame(gps_array, columns=['datetime', 'speed'])
    gps_df = gps_df.sort_values(by='datetime')
    gps_array = gps_df.values
    speed = gps_df.speed
    zero_velocity = (speed==0).astype(int)
    stopped_ids = np.where(np.diff(zero_velocity,prepend=0)==1)[0] # vehicle stopped event is on
    started_ids = np.where(np.diff(zero_velocity, append=0)==-1)[0] # vehicle started, event is off
    starts = gps_array.T[0][started_ids] # started to drive with speed>0 (end of event)
    ends = gps_array.T[0][stopped_ids] # stopped to drive, velocity ==0 (begin of event)
    duration = starts-ends
    pause_event = duration>timedelta(seconds=60) # event whose stop is considered pause in operation
    events = pd.DataFrame(np.column_stack([stopped_ids,started_ids,starts,ends,duration,pause_event]),
                          columns = ['stopped_ids', 'started_ids', 'starts', 'ends', 'duration','pause_event'])
    continous_events = []
    for i,e in events.iterrows():
        continous = is_event_continous(e,  gps_df) # event is considered continous if signal is not stopping for more that one minute.
        print(f"duration: {e.duration}, continous: {continous}")
        continous_events.append(continous)
    events['continous_events'] = continous_events

    # worked = pd.Series(["worked"]*gps_df.shape[0])
    # for (i, e) in events.iterrows():
    #     if not e.continous_events:
    #         worked[e.stopped_ids:e.started_ids] = 'off'
    #     elif e.pause_event:
    #         worked[e.stopped_ids:e.started_ids] = 'pause'
    #
    # for work_status in ['off', 'pause', 'worked']:
    #     speed = gps_df[worked==work_status].speed
    #     d = gps_df[worked==work_status].datetime
    #     plt.scatter(d,speed, label=work_status)
    # plt.legend()
    # plt.show()
    return gps_df,events



# gps_array = np.load("/media/backup/Algo/users/avinoam/safety_analysis/yavne_speed.npy", allow_pickle=True)[1:]
gps_array = speed_array
gps_df = pd.DataFrame(gps_array, columns=['datetime', 'speed'])
gps_df = gps_df.sort_values(by='datetime')

gps_df, events = gps_array_to_stop_events(gps_df)

worked = pd.Series(["worked"]*gps_df.shape[0])
for (i, e) in events.iterrows():
    if not e.continous_events:
        worked[e.stopped_ids:e.started_ids] = 'off'
    elif e.pause_event:
        worked[e.stopped_ids:e.started_ids] = 'pause'

for work_status in ['worked', 'off', 'pause']:
    speed = (gps_df[worked==work_status]).speed
    d = gps_df[worked==work_status].datetime
    plt.scatter(d,speed, label=work_status, s=1)
plt.scatter(gps_array.T[0][gps_array.T[1]==None],np.zeros_like(gps_array.T[0][gps_array.T[1]==None]), label="missing_data")
plt.legend()
plt.show()

gps_df['date'] = gps_df['datetime'].apply(lambda dt:dt.date())
worked_by_day = gps_df[worked=='worked']['date'].value_counts().sort_index()
pause_by_day = gps_df[worked=='pause']['date'].value_counts().sort_index()
off_by_day = gps_df[worked=='off']['date'].value_counts().sort_index()

start_date = min((worked_by_day.index[0],pause_by_day.index[0],off_by_day.index[0]))
end_date = max((worked_by_day.index[-1],pause_by_day.index[-1],off_by_day.index[-1]))

all_dates = np.arange(start_date, end_date)
all_dates = np.append(all_dates, end_date)

# add zeros were there is missing data, to enable bars bottom
for work_status in [worked_by_day, pause_by_day, off_by_day]:
    for d in all_dates:
        if d not in work_status.index.values:
            work_status.loc[d] = 0

pause_by_day = pause_by_day.sort_index()
worked_by_day = worked_by_day.sort_index()
off_by_day = off_by_day.sort_index()

plt.bar(x=pause_by_day.index, height=pause_by_day, label='pause')
plt.bar(x=worked_by_day.index, height=worked_by_day, label='work', bottom=pause_by_day)
plt.bar(x=off_by_day.index, height=off_by_day, label='parking', bottom=pause_by_day+worked_by_day)
plt.legend()
plt.show()
# np.save("/media/backup/Algo/users/avinoam/safety_analysis/yavne_speed.npy" , speed_array)
datetime2matplottime  = lambda d : dates.datetime.time(d.hour,d.minute,d.second,d.microsecond)
datetime2time = lambda d : datetime.combine(datetime.now(),d.time())
times = [datetime2matplottime(d) if type(d) != type(None) else None for d in gps_array.T[0]]
hours = [d.hour + d.minute / 60 for d in gps_array.T[0] if type(d) != type(None)]
hours_bins = np.histogram(hours, bins = np.arange(0,25,1))
events_hours_bins = np.histogram(events.hour , bins = np.arange(0,25,1))
# plt.scatter(hours_bins[1][1:],events_hours_bins[0]/hours_bins[0], label='all')
plt.scatter(hours_bins[1][1:],events_hours_bins[0]/hours_bins[0]*3600, label='all')

ids = hours_bins[0]>=5
# fit = np.polyfit(1/events_hours_bins[1][1:][ids],events_hours_bins[0][ids]/hours_bins[0][ids], deg=3)
# plt.plot(events_hours_bins[1][1:][ids], np.polyval(fit,1/(events_hours_bins[1][1:][ids])))
#
# for severity in [1,2,3]:
#     events_hours_bins = np.histogram(events[events.Severity==severity].hour , bins = np.arange(0,25,1))
#     ids = hours_bins[0]>=5
#     # fit = np.polyfit(1/events_hours_bins[1][1:][ids],events_hours_bins[0][ids]/hours_bins[0][ids], deg=3)
#     # plt.plot(events_hours_bins[1][1:][ids], np.polyval(fit,1/(events_hours_bins[1][1:][ids])))
#     print("events counter",events_hours_bins[0])
#     print("hours counter",hours_bins[0])
#     print("bins-boundaries",hours_bins[1])
#     plt.scatter(hours_bins[1][1:],events_hours_bins[0]/hours_bins[0]*3600, label=str(severity))
# plt.legend()
# plt.show()
# times = [datetime.combine(datetime.now(),d.time()) if type(d) != type(None) else None for d in speed_array.T[0] ]
plt.subplot(1,2,1)
plt.scatter(gps_array.T[0], gps_array.T[1])
plt.plot(gps_array.T[0], gps_array.T[1])

plt.subplot(1,2,2)
# plt.scatter(times,speed_array.T[1])
# plt.show()
plt.plot(gps_array.T[0], gps_array.T[1])
plt.title("YAVNE-DL250")
plt.xlabel("time")
plt.ylabel("velocity [kph]")
plt.show()


pass