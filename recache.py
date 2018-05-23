import os
import json
import argparse
import subprocess
from string import Template
import logging


def load_datasets_manifest(file_obj, environ=None):
    ''' Read in a datasets manifest file and substitute environment variables
    '''
    
    if environ is None:
        environ = os.environ

    template = Template(file_obj.read())
    template = template.substitute(**environ)

    return json.loads(template)


def process_dataset(dataset, dirname, base_path, recache):

    data_dir = os.path.join(dirname, dataset['data-dir'])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_paths = [ os.path.join(data_dir, fil['path']) for fil in dataset['files'] ]
    existing_files, missing_files_str = check_existing_files(file_paths)

    if recache:
        logging.info('force recaching dataset {}'.format(dataset['name']))
        recache_dataset(dataset, base_path, file_paths)
    
    elif dataset['auto-generate'] and sum(existing_files.values()) == 0:
        logging.info('files for dataset {} don\'t exist, regenerating'.format(dataset['name']))
        recache_dataset(dataset, base_path, file_paths)

    elif dataset['auto-generate'] and sum(existing_files.values()) > 0:
        logging.warning('dataset {} missing files at: {}. Consider recaching '.format(dataset['name'], missing_files_str))
    
    else:
        logging.info('all files for dataset {} exist, skipping'.format(dataset['name']))


def recache_dataset(dataset, base_path, file_paths):

    command = [dataset['script']] + dataset['arguments']
    logging.info(' '.join(command))
    subprocess.check_call(command, cwd=base_path)

    existing_files, missing_files_str = (file_paths, check=os.path.isfile) # not sure why isfile here vs. exists
    if not all(existing_files.values()):
        raise RuntimeError('files missing after generating dataset {}: {}'.format(dataset['name'], missing_files_str))


def check_existing_files(file_paths, check=os.path.exists):

    existing_files = { file_path: check(file_path) for file_path in file_paths }
    missing_files_str = '\n'.join([ path for path, exists in existing_files.items() if not exists ])

    return existing_files, missing_files_str


def main(dataset_manifests, recache):

    for datasets_json in args.dataset_manifests:
        datasets = load_datasets_manifest(datasets_json)

        datasets_dirname = os.path.dirname(datasets_json.name)
        base_path = os.path.dirname(os.path.abspath(datasets_json.name))

        for dataset in datasets:
            if not dataset['enabled']:
                logging.info('skipping disabled dataset {}'.format(dataset['name']))
                continue

            process_dataset(dataset, datasets_dirname, base_path, recache)


if __name__ == '__main__':
    ''' Generate any data files in specified manifest that don't already exist, or force all
    to regenerate and overwrite any existing data files using --recache or by setting the
    environment variable DATACUBE_RECACHE '''

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_manifests', type=argparse.FileType('r'), nargs='+', help='JSON dataset manifest files')
    parser.add_argument('--recache', action='store_true', help='regenerate all data files and exit', default=('DATACUBE_RECACHE' in os.environ))
    
    args = parser.parse_args()
    main(args.dataset_manifests, args.recache)
