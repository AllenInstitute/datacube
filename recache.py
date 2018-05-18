import os
import json
import argparse
import subprocess
from string import Template

if __name__ == '__main__':
    ''' Generate any data files in specified manifest that don't already exist, or force all
    to regenerate and overwrite any existing data files using --recache or by setting the
    environment variable DATACUBE_RECACHE '''

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_manifests', type=argparse.FileType('r'), nargs='+', help='JSON dataset manifest files')
    parser.add_argument('--recache', action='store_true', help='regenerate all data files and exit', default=('DATACUBE_RECACHE' in os.environ))
    args = parser.parse_args()

    for datasets_json in args.dataset_manifests:
        datasets = json.loads(Template(datasets_json.read()).substitute(**os.environ))
        for dataset in datasets:
            if dataset['enabled']:
                data_dir = os.path.join(os.path.dirname(datasets_json.name), dataset['data-dir'])

                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                existing = [os.path.isfile(os.path.join(data_dir, f['path'])) for f in dataset['files']]
                if args.recache or (dataset['auto-generate'] and sum(existing) == 0):
                    command = [dataset['script']] + dataset['arguments']
                    print(' '.join(command))
                    basepath = os.path.dirname(os.path.abspath(datasets_json.name))
                    subprocess.check_call(command, cwd=basepath)
                    existing = [os.path.isfile(os.path.join(data_dir, f['path'])) for f in dataset['files']]
                    if not all(existing):
                        raise RuntimeError('Expected file(s) missing after running \'{}\' script.'.format(dataset['name']))
                else:
                    print('file from dataset \'{}\' (\'{}\') exists; skipping...'.format(dataset['name'], datasets_json.name))
