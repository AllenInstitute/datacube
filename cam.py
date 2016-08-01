import pg8000
import h5py
import numpy as np
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet

def load():
    conn = pg8000.connect(user='postgres', host='localhost', port=5432, database='lims', password='postgres')
    cursor = conn.cursor()
    cursor.execute("""
        select distinct ec.id from experiment_containers ec
        join experiment_sessions es on ec.id = es.experiment_container_id
        where es.workflow_state = 'passed'
        order by ec.id
    """)
    experiment_container_ids = cursor.fetchall()
    
    row_ids = []
    n_ns = 119
    n_sg = (6*6*4)
    n_dg = (8*6)
    num_columns = n_ns + n_sg + n_dg
    all_sessions = []
    
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
        
        row_ids += merged_cs_ids
    
    row_ids.sort()
    A = np.full((len(row_ids), num_columns, 3), np.nan)
    
    for session in all_sessions:
        f = h5py.File(session[0],'r')
        print 'Reading %s' % session[0]
        
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

    return A
