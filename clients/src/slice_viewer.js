import autobahn from "autobahn";

export class DatacubeConnection {
    constructor(url, realm, onopen) {
        var self = this;
        self.connection = new autobahn.Connection({
             url: url,
             realm: realm
        });

        self.session = null;
        self.connection.onopen = function (s) {
            self.session = s;
            onopen(self);
        };

        self.connection.open();
    }

    call(method, args, kwargs) {
        var self = this;
        return self.session.call(method, args, kwargs);
    }
}

export class Datacube {
    constructor(datacube_connection, datacube_name, onload) {
        var self = this;
        self.conn = datacube_connection;
        self.name = datacube_name;
        self.info = null;
        
        var call = 'org.brain-map.api.datacube.info';
        self.call(call, [], {}).then(function (res) {
            self.info = res;
            onload(self);
        });
    }

    call(method, args, kwargs) {
        var self = this;
        return self.conn.call(method + '.' + self.name, args, kwargs);
    }
}

export class DatacubeSliceView {
    //todo: make images dynamically instead of passing in
    constructor(datacube, dims, orientations, flips, fields, field_kwargs, canvases, controls, images, render_callback, resize_callback, coords_callback) {
        var self = this;
        self.datacube = datacube;
        self.fields = fields;
        self.field_kwargs = field_kwargs;
        self.dims = dims;
        self.orientations = orientations;
        self.flips = flips;
        self.render_callback = render_callback;
        self.resize_callback = resize_callback;
        self.coords_callback = coords_callback;
        self.canvases = canvases;
        self.controls = controls;
        self.images = images;
        self.pending = {};
        self.did_fire = {};
        self.coords = {};
        dims.forEach(function (dim) {
            self.pending[dim] = false;
            self.did_fire[dim] = false;
        });

        self.dims.forEach(function (dim) {
            function scroll_handler() {
                self.dims.forEach(function (dim) {
                    if(self.did_fire[dim]) {
                        if(!self.pending[dim]) {
                            self.coords[dim] = self.controls[dim].get_current_value();
                            self.update();
                        }
                    }
                });
                self.coords_callback(self.coords);
                self.did_fire[dim] = true;
            }
            var initial = Math.round(self.datacube.info.dims[dim]/2);
            self.coords[dim] = initial;
            self.controls[dim].setup(0,
                                     self.datacube.info.dims[dim]-1,
                                     initial,
                                     function(event, ui) {
                                        scroll_handler();
                                     });
        });
        self.update();
        self.coords_callback(self.coords);

        setInterval(function() {
            self.dims.forEach(function (dim) {
                self.did_fire[dim] = false;
            });
        }, 50);

        for(let click_dim in self.canvases) {
            function coord_handler(event) {
                var canvas = event.target;
                var rect = canvas.getBoundingClientRect();
                var x = Math.round(((event.clientX - rect.left)/canvas.width)*self.datacube.info.dims[self.orientations[click_dim][1]]);
                var y = Math.round(((event.clientY - rect.top)/canvas.height)*self.datacube.info.dims[self.orientations[click_dim][0]]);
                var idx = self.coords[click_dim];
                var coords = {};
                coords[self.orientations[click_dim][1]] = x;
                coords[self.orientations[click_dim][0]] = y;
                coords[click_dim] = idx;
                self.dims.forEach(function (dim) {
                    if (-1 == self.flips[click_dim][dim]) {
                        coords[dim] = self.datacube.info.dims[dim] - coords[dim];
                    }
                });
                self.coords = coords;
                self.coords_callback(coords);
                //todo: maybe don't update the dim that was clicked
                self.update();
            }
            $(self.canvases[click_dim]).click(coord_handler);
            var did_drag = false;
            $(self.canvases[click_dim]).mousemove(function (event) {
                if(1 == event.buttons) {
                    if(did_drag) {
                        var any_pending = true;
                        self.dims.forEach(function (dim) {
                            any_pending = any_pending & self.pending[dim];
                        });
                        if(!any_pending) {
                            coord_handler(event);
                        }
                    }
                    did_drag = true;
                }
            });
            setInterval(function() {
                did_drag = false;
            }, 50);
        }
    }

    update() {
        var self = this;
        self.dims.forEach(function (dim) {
            self.update_dim(dim);
        });
    }

    update_dim(dim) {
        var self = this;
        self.pending[dim] = true;
        var val = self.coords[dim];
        var finished = {};
        function finish() {
            var all_finished = true;
            self.fields.forEach(function (field) {
                all_finished = all_finished & finished[field];
            });
            if(self.pending[dim] && all_finished) {
                self.render(dim);
                self.pending[dim] = false;
            }
        }

        self.fields.forEach(function (field, i) {
            finished[field] = false;
            var select = {};
            self.datacube.info.vars[field].dims.forEach(function (field_dim) {
                if(field_dim == dim) {
                    select[field_dim] = val; //todo: flip
                } else {
                    select[field_dim] = {'start': null, 'stop': null, 'step': self.flips[dim][field_dim]};
                }
            });
            var call = 'org.brain-map.api.datacube.image';
            var args = [];
            var kwargs = Object.assign({'field': field, 'select': select, 'dim_order': self.orientations[dim]}, self.field_kwargs[dim][field]);
            self.datacube.call(call, args, kwargs).then(
                function (res) {
                    if(res.data) {
                        var image = self.images[dim][field];
                        image.onload = function() {
                            finished[field] = true;
                            finish(field);
                        };
                        image.src = res.data;
                    } else {
                        //todo: error
                    }
                }
            );
        });
    }

    render(dim) {
        var self = this;
        var canvas = self.canvases[dim];
        var images = self.images[dim];
        self.resize_callback(self, canvas, images);
        var xdim = self.orientations[dim][1];
        var ydim = self.orientations[dim][0];
        var x = self.coords[xdim]/self.datacube.info.dims[xdim];
        var y = self.coords[ydim]/self.datacube.info.dims[ydim];
        if (-1 == self.flips[dim][xdim]) {
            x = 1.0-x;
        }
        if (-1 == self.flips[dim][ydim]) {
            y = 1.0-y;
        }
        var x = Math.floor(x*canvas.width);
        var y = Math.floor(y*canvas.height);
        self.render_callback(self, x, y, canvas, images);
    }

    redraw() {
        var self = this;
        self.dims.forEach(function (dim) {
            self.render(dim);
        });
    }
}

