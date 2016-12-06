import pg8000
import h5py
import json
import zlib
import base64
import numpy as np
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
from progressbar import ProgressBar, Percentage, Bar, ETA, Counter, FileTransferSpeed
from itertools import product

def load():
    # TODO: pass in connection params
    conn = pg8000.connect(user='atlasreader', host='limsdb2', port=5432, database='lims2', password='atlasro')
    cursor = conn.cursor()
    cursor.execute("""
        select distinct ec.id from experiment_containers ec
        where ec.published_at IS NOT NULL
        order by ec.id
    """)
    experiment_container_ids = cursor.fetchall()
    
    row_ids = []
    all_sessions = []
    ns_frames = []
    sg_orivals = []
    sg_sfvals = []
    sg_phasevals = []
    dg_orivals = []
    dg_tfvals = []
    
    for ec_id in experiment_container_ids:
        cursor.execute("""
            select wkf.storage_directory || wkf.filename as h5file,
            wkf2.storage_directory || wkf2.filename as nwbfile,
            substring(wkf.filename from '([ABC])_analysis.h5$') as session
            from experiment_sessions es
            join well_known_files wkf on wkf.attachable_id = es.id
            join well_known_files wkf2 on wkf2.attachable_id = es.id
            join experiment_containers ec on ec.id = es.experiment_container_id
            where es.workflow_state = 'passed'
            and ec.id = """ + str(ec_id[0]) + """
            and wkf.filename ilike '%%analysis.h5%%'
            and wkf2.filename ilike '%%.nwb%%'
            order by substring(wkf.filename from '([ABC])_analysis.h5$')
        """)
        sessions = cursor.fetchall()
        all_sessions += sessions
    
        cs_ids = dict()
        merged_cs_ids = []
        
        for session in sessions:
            f = h5py.File(session[0],'r')
    
            nwb_file = session[1]
            data_set = BrainObservatoryNwbDataSet(nwb_file)
            cs_ids[session[2]] = data_set.get_cell_specimen_ids()
            merged_cs_ids = list(set().union(merged_cs_ids, cs_ids[session[2]]))

            try:
                stimulus_table = data_set.get_stimulus_table('natural_scenes')
                stim_table = stimulus_table.fillna(value=0.)
                ns_frames = list(set().union(ns_frames, np.unique(stim_table.frame).astype(int)))
            except:
                pass

            try:
                stimulus_table = data_set.get_stimulus_table('static_gratings')
                stim_table = stimulus_table.fillna(value=0.)
                sg_orivals = list(set().union(sg_orivals, np.unique(stim_table.orientation.dropna())))
                sg_sfvals = list(set().union(sg_sfvals, np.unique(stim_table.spatial_frequency.dropna())))
                sg_phasevals = list(set().union(sg_phasevals, np.unique(stim_table.phase.dropna())))
            except:
                pass

            try:
                stimulus_table = data_set.get_stimulus_table('drifting_gratings')
                stim_table = stimulus_table.fillna(value=0.)
                dg_orivals = list(set().union(dg_orivals, np.unique(stim_table.orientation).astype(int)))
                dg_tfvals = list(set().union(dg_tfvals, np.unique(stim_table.temporal_frequency).astype(int)))
            except:
                pass
        
        row_ids += merged_cs_ids
    

    row_meta_fields = ["cell_specimen_id", "experiment_container_id", "area", "tld1_id", "tld1_name",
        "tld2_id", "tld2_name", "tlr1_id", "tlr1_name", "imaging_depth", "osi_dg", "dsi_dg",
        "pref_dir_dg", "pref_tf_dg", "p_dg", "osi_sg", "pref_ori_sg", "pref_sf_sg", "pref_phase_sg",
        "p_sg", "time_to_peak_sg", "pref_image_ns", "time_to_peak_ns", "p_ns", "all_stim"]

    wh_conn = pg8000.connect(user='postgres', host='testwarehouse1', port=5432, database='warehouse-R193', password='postgres')
    wh_cursor = wh_conn.cursor()
    with open('./data/cam_rows.json.zz.b64', 'w') as json_file:
        wh_cursor.execute('select ' + ', '.join(row_meta_fields) + ' from api_cam_cell_metrics order by cell_specimen_id asc')
        #json_file.write(json.dumps([row_meta_fields] + list(wh_cursor.fetchall()), json_file))
        s = json.dumps([{row_meta_fields[i]: v for i,v in enumerate(row)} for row in list(wh_cursor.fetchall())])
        json_file.write(base64.b64encode(zlib.compress(s)))


    n_ns = len(ns_frames)
    n_sg = len(sg_orivals)*len(sg_sfvals)*len(sg_phasevals)
    n_dg = len(dg_orivals)*len(dg_tfvals)
    num_columns = n_ns + n_sg + n_dg

    col_meta_fields = ["stimulus_type", "frame", "orientation", "spatial_frequency", "phase", "temporal_frequency"]
    col_meta = []
    for frame in ns_frames:
        col_meta.append({"stimulus_type": "ns", "frame": frame})
    for stim in product(sg_orivals, sg_sfvals, sg_phasevals):
        col_meta.append({"stimulus_type": "sg", "orientation": float(stim[0]), "spatial_frequency": float(stim[1]), "phase": float(stim[2])})
    for stim in product(dg_orivals, dg_tfvals):
        col_meta.append({"stimulus_type": "dg", "orientation": stim[0], "temporal_frequency": stim[1]})

    with open('./data/cam_cols.json.zz.b64', 'w') as json_file:
        json_file.write(base64.b64encode(zlib.compress(json.dumps(col_meta))))


    row_ids.sort()
    A = np.full((len(row_ids), num_columns, 3), np.nan)

    progress = ProgressBar(widgets=[Percentage(), ' ', Bar(), ' ', Counter(), '/' + str(len(all_sessions)) + ' nwb files ', ETA(), ' ', FileTransferSpeed(unit='files')], maxval=len(all_sessions))
    progress.start()
    for session_index, session in enumerate(all_sessions):
        f = h5py.File(session[0],'r')
        
        nwb_file = session[1]
        data_set = BrainObservatoryNwbDataSet(nwb_file)
        cs_ids = data_set.get_cell_specimen_ids()
        
        for idx, cs_id in enumerate(cs_ids):
            row_id = np.searchsorted(row_ids, cs_id)
    
            if 'response_ns' in f['analysis'].keys():
                A[row_id,:n_ns,:] = f['analysis']['response_ns'][:,idx,:].reshape((n_ns, 3))
            if 'response_sg' in f['analysis'].keys():
                A[row_id,n_ns:(n_ns+n_sg),:] = f['analysis']['response_sg'][:,:,:,idx,:].reshape((n_sg, 3))
            if 'response_dg' in f['analysis'].keys():
                A[row_id,(n_ns+n_sg):,:] = f['analysis']['response_dg'][:,:,idx,:].reshape((n_dg, 3))

        progress.update(session_index)
    progress.finish()

    return A
