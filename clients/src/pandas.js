import autobahn from "autobahn";
import pako from "pako";
import {StructuredArray} from "./common/structured_array.js";
import {ScrollPreloader} from "./common/scroll_preloader.js";


export function PandasClient(wamp_router_url,
                             wamp_realm,
                             datacube_name,
                             chunk_size,
                             preload_margin,
                             get_current_filters,
                             first_page_callback,
                             apply_filters_callback,
                             get_current_page_range,
                             page_data_callback,
                             update_finished) {
    this.datacube_name = datacube_name;
    this.connection = new autobahn.Connection({url: wamp_router_url, realm: wamp_realm});
    this.session = null;

    this.scroll_preloader = new ScrollPreloader(100,
                                                25,
                                                this.load_indexes.bind(this),
                                                this.load_records.bind(this),
                                                get_current_filters,
                                                first_page_callback,
                                                apply_filters_callback,
                                                get_current_page_range,
                                                page_data_callback);
    this.update_finished = update_finished;

    var self = this;
    this.connection.onopen = function (s) {
        self.session = s;
        self.scroll_preloader.apply_filters();
    };

    this.connection.open();
};


PandasClient.prototype.apply_filters = function() {
    this.scroll_preloader.apply_filters();
};


PandasClient.prototype.update_page = function() {
    this.scroll_preloader.update_page(this.update_finished);
};


PandasClient.prototype.load_indexes = function(kwargs_in, start, stop, callback) {
    var kwargs = {};
    kwargs.name = this.datacube_name;
    kwargs.filters = kwargs_in.filters;
    kwargs.sort = kwargs_in.sort;
    kwargs.ascending = kwargs_in.ascending;
    kwargs.fields = 'indexes_only';
    if(start != null) {
        kwargs.start = start;
    }
    if(stop != null) {
        kwargs.stop = stop;
    }
    this.session.call('org.alleninstitute.pandas_service.filter_cell_specimens', [], kwargs).then(
        function (res) { callback(res.indexes, res.filtered_total); });
};


PandasClient.prototype.load_records = function(indexes, range, callback) {
    if(range.end <= range.start) {
        callback([], range);
    } else {
        var indexes = indexes.slice(range.start, range.end);
        this.session.call('org.alleninstitute.pandas_service.filter_cell_specimens', [],
                         {'indexes': indexes, 'fields': JSON.parse(document.getElementById('fields').value), 'name': this.datacube_name}).then(
            function (res) {
                //var zbuf = atob(res.data);
                var zbuf = res.data;
                var buf = pako.inflate(zbuf);
                res.data = buf;
                var sa = new StructuredArray(res);
                var records = sa.lazy_get_all();
                callback(records, range);
            }
        );
    }
};


