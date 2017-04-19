import autobahn from "autobahn";
import pako from "pako";
import {StructuredArray} from "./common/structured_array.js";
import {ScrollPreloader} from "./common/scroll_preloader.js";


var url = 'ws://' + window.location.hostname + ':8081/ws';
var realm = 'aibs';
var connection = new autobahn.Connection({url: url, realm: realm});
var session = null;


function load_indexes(kwargs_in, start, stop, callback) {
    var kwargs = {};
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
    session.call('org.alleninstitute.pandas_service.filter_cell_specimens', [], kwargs).then(
        function (res) { callback(res.indexes, res.filtered_total); });
};


function load_records(indexes, range, callback) {
    if(range.end <= range.start) {
        callback([], range);
    } else {
        var indexes = indexes.slice(range.start, range.end);
        session.call('org.alleninstitute.pandas_service.filter_cell_specimens', [],
                         {'indexes': indexes, 'fields': JSON.parse(document.getElementById('fields').value)}).then(
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


export function PandasClient(chunk_size,
                             preload_margin,
                             get_current_filters,
                             first_page_callback,
                             apply_filters_callback,
                             get_current_page_range,
                             page_data_callback,
                             update_finished) {
    self.scroll_preloader = new ScrollPreloader(100,
                                                25,
                                                load_indexes,
                                                load_records,
                                                get_current_filters,
                                                first_page_callback,
                                                apply_filters_callback,
                                                get_current_page_range,
                                                page_data_callback);
    self.update_finished = update_finished;
};


PandasClient.prototype.apply_filters = function() {
    self.scroll_preloader.apply_filters();
};


PandasClient.prototype.update_page = function() {
    self.scroll_preloader.update_page(self.update_finished);
};


connection.onopen = function (s) {
    session = s;
    scroll_preloader.apply_filters();
};


connection.open();
