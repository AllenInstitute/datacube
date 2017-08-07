import json

class NeuronLoader():
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        points = dict()

        with open(self.file_path) as f:

            for line in f:

                if '#' in line:
                    continue

                line = line.strip().split(' ')
                
                point = {
                    'id':       int(line[0]),
                    'type':     int(line[1]),
                    'x':        float(line[2]),
                    'y':        float(line[3]),
                    'z':        float(line[4]),
                    'r':        float(line[5]),
                    'pid':      int(line[6]),
                    'children': []
                }

                if point['id'] not in points:
                    points.setdefault(point['id'], point)
                else:
                    point['children'] = points[point['id']]['children']
                    points[point['id']] = point
                
                if point['pid'] not in points:
                    points.setdefault(point['pid'], { 'children': [ point ] })
                else:
                    points[point['pid']]['children'].append(point)
        
        return points[1]







        