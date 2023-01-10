import logging

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from sklearn.cluster import DBSCAN

from utils.coordinates_convertor import CoordinateConvector
from utils.read_events_features import EventDataMining_new


im = plt.imread('/media/backup/Algo/users/avinoam/safety_analysis/Modiim/map/Govmap.png')
pgw_fn = '/media/backup/Algo/users/avinoam/safety_analysis/Modiim/map/Govmap.pgw'
extent = CoordinateConvector().convert_pgw_ITM_to_extent(pgw_fn, im)
# CoordinateConvector().from_ITM_to_WGS84()
def position_hist_2d(events):
    # 2d histogram over position
    positions = events[['lon','lat']].values
    lon, lat = positions[:,0], positions[:,1]

    position_hist = np.histogram2d(lon,lat, bins=5)
    H, lon_edges, lat_edges = position_hist

    return H, lon_edges, lat_edges

def position_clustering_dbscan(events):
    positions = events[['lon','lat']].values
    # eps = np.percentile(np.absolute(np.diff(positions.T.flatten())),50)
    eps = 0.0005
    clustering = DBSCAN(eps=eps, min_samples=2).fit(positions)
    return clustering.labels_
    pass

def clustering_dbscan(events):
    features = events[['lon','lat','hour', 'speed']].values
    # eps = np.percentile(np.absolute(np.diff(positions.T.flatten())),50)
    for i in range(features.shape[1]-1): # we do not normalize speed lineary, but log-speed
        mean = features.T[i].mean()
        std = features.T[i].std()
        features.T[i] = (features.T[i]-mean)/std

    def linear_norm(x):
        max = x.max()
        mean = x.mean()
        return (x-mean)/(max-mean)

    def gaussian_norm(x):
        mean = x.mean()
        std = x.std()
        return (x-mean)/std



    f = features.T[-1]
    # plt.subplot(1,1,1)
    # plt.scatter(f,linear_norm(f), label='linear_norm')
    # # plt.scatter(f,linear_norm(np.log(f+1)), label='linear_norm: log(speed+1)')
    # plt.scatter(f,gaussian_norm(f), label= 'gaussian_norm')
    # plt.xlabel('speed [kph]')
    # plt.ylabel('normalized')
    #
    # plt.legend()
    # plt.show()
    features.T[i] = linear_norm(f)

    # plt.scatter(f,gaussian_norm(f), label='gaussian')
    # plt.scatter(f,linear_norm(f), label='linear_norm')


    eps = 0.5
    clustering = DBSCAN(eps=eps, min_samples=2).fit(features)
    return clustering.labels_
    pass


# --- CONSTS ----
MODIIM_MAP_FN =  '/media/backup/Algo/users/avinoam/safety_analysis/Modiim/map/Govmap.png'
# MODIIM_EVENTS_CSV_FN = '/media/backup/Algo/users/avinoam/safety_analysis/Modiim/csv_file_2023_01_08_10_03_20.csv'
MODIIM_EVENTS_CSV_FN = '/media/backup/Algo/users/avinoam/safety_analysis/Modiim/csv_file_2023_01_09_11_28_05.csv'
MODIIM_EVENTS_DF_FN = '/media/backup/Algo/users/avinoam/safety_analysis/Modiim/modiin_df.csv'



if __name__ == "__main__":
    EXTRACT_EVENTS_NEW = False
    if EXTRACT_EVENTS_NEW:
        events_file = MODIIM_EVENTS_CSV_FN
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
        events_df.to_csv(MODIIM_EVENTS_DF_FN)
    else:
        events_df = pd.read_csv(MODIIM_EVENTS_DF_FN)

    map_fn = MODIIM_MAP_FN
    map = plt.imread(map_fn)

    PLOT_EVENTS_HISTOGRAM = False
    if PLOT_EVENTS_HISTOGRAM:
        events_df = events_df[:-1]
        plt.imshow(map, extent = extent, aspect='equal')
        H, lon_edges, lat_edges = position_hist_2d(events_df)
        X, Y = np.meshgrid(lon_edges, lat_edges)
        bounds = [0,1,2,3,5,10,20,30]
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)
        hist2d = H.T
        hist2d = np.ma.masked_where(hist2d < 1, hist2d)
        plt.imshow(hist2d, extent=(lon_edges[0], lon_edges[-1], lat_edges[-1], lat_edges[0]), alpha=0.5, origin ='upper',
                   aspect='equal', norm=norm, cmap='Reds')#plt.cm.get_cmap('Reds', H.max()+1))
        plt.colorbar()
        df = events_df
        plt.scatter(df.lon, df.lat, alpha=0.8)
        plt.xlabel('Lon [deg]')
        plt.ylabel('Lat [deg]')
        plt.title('Geo-Grid density of events')
        plt.show()

    PLOT_TIME_HISTOGRAM = True
    if PLOT_TIME_HISTOGRAM:
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

    PLT_DBSCAN_CLUSTERS = False
    if PLT_DBSCAN_CLUSTERS:
        if True:
            plt.imshow(map, extent = extent, aspect='equal')
            plt.xlabel('Lon [deg]')
            plt.ylabel('Lat [deg]')
            plt.title('DB-scan clustering of events')
            labels = position_clustering_dbscan(events_df)
            # colors = np.random.randint(0,225,(max(labels)+1,3))/255
            colors = mcolors.BASE_COLORS
            df = events_df[labels!=-1]
            c = colors[labels[labels!=-1]]
            plt.scatter(df.lon, df.lat, alpha=0.8, c = labels[labels!=-1], cmap=plt.cm.get_cmap('cubehelix', labels.max()+1))
            df = events_df[labels==-1]
            c = np.array((20,100,100))/255
            plt.scatter(df.lon, df.lat, alpha=0.8, c=c, marker='*', s=15)
            c = np.array((200,100,100))/255
            plt.scatter(df.lon, df.lat, alpha=0.8, c=c, marker='*', s=5)
            plt.show()



    PLT_DBSCAN_CLUSTERS_ND = True
    if PLT_DBSCAN_CLUSTERS_ND: # N-dimensions
        def on_zoom_change(ax):
            y_ticks = plt.gca().get_yticks()
            plt.gca().set_yticklabels(['{:.3f}'.format(y) for y in y_ticks])
            x_ticks = plt.gca().get_xticks()
            plt.gca().set_xticklabels(['{:.3f}'.format(x) for x in x_ticks])
            pass
        ax = plt.subplot(1,2,1)
        plt.imshow(map, extent = extent, aspect='equal')
        plt.xlabel('Lon [deg]')
        plt.ylabel('Lat [deg]')
        plt.title('DB-scan clustering of events')
        labels = clustering_dbscan(events_df)
        # colors = np.random.randint(0,225,(max(labels)+1,3))/255
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()
        colors = np.array(list(colors))
        df = events_df[labels!=-1]
        c = colors[labels[labels!=-1]]
        # plt.scatter(df.lon, df.lat, alpha=1.0, c=labels[labels!=-1], cmap=plt.cm.get_cmap('cubehelix', labels.max()+1))
        plt.scatter(df.lon, df.lat, alpha=1.0, c=c)
        plt.legend()
        df = events_df[labels==-1]
        plt.scatter(df.lon, df.lat, alpha=1.0, c=colors[-4], marker='*')
        plt.scatter(df.lon, df.lat, alpha=1.0, c='r', marker='.', s=5)

        ax.callbacks.connect('xlim_changed', on_zoom_change)
        ax.callbacks.connect('ylim_changed', on_zoom_change)

        df = events_df[labels!=-1]

        ax = plt.subplot(1,2,2)
        def on_zoom_change(ax):
            y_ticks = plt.gca().get_yticks()
            plt.gca().set_yticklabels(['{:.1f}'.format(y) for y in y_ticks])
            x_ticks = plt.gca().get_xticks()
            plt.gca().set_xticklabels(['{:.1f}'.format(x) for x in x_ticks])
            pass
        # plt.scatter(df.speed, df.hour, c=labels[labels!=-1], cmap=plt.cm.get_cmap('cubehelix', labels.max()+1))
        c = colors[labels[labels!=-1]]
        plt.scatter(df.speed, df.hour, c=c)
        df = events_df[labels==-1]
        plt.scatter(df.speed, df.hour, alpha=1.0, c=colors[-4], marker='*')
        plt.scatter(df.speed, df.hour, alpha=1.0, c='r', marker='.', s=5)
        plt.xlabel('speed [kph]')
        plt.ylabel('hour of day')
        ax.callbacks.connect('xlim_changed', on_zoom_change)
        ax.callbacks.connect('ylim_changed', on_zoom_change)
        on_zoom_change(ax)
        ax.set_facecolor((0.7,0.7,0.8))

        plt.show()
        pass