'use strict';

export function ScrollPreloader(chunk_size,
                                preload_margin,
                                load_indexes,
                                load_records,
                                get_current_filters,
                                first_page_callback,
                                apply_filters_callback,
                                get_current_page_range,
                                page_data_callback) {

    this.chunk_size = chunk_size;
    this.preload_margin = preload_margin;

    this.load_indexes = load_indexes;
    this.load_records = load_records;
    this.get_current_filters = get_current_filters;
    this.first_page_callback = first_page_callback;
    this.apply_filters_callback = apply_filters_callback;
    this.get_current_page_range = get_current_page_range;
    this.page_data_callback = page_data_callback;

    this.previous_range = {"start": 0, "end": 0};
    this.records = [];
};


ScrollPreloader.prototype.apply_filters = function() {
    var kwargs = this.get_current_filters();
    var self = this;
    this.load_indexes(kwargs, 0, 2 * this.chunk_size, function (indexes, total) {
            self.indexes = indexes;
            self.previous_range = {"start": 0, "end": 0};
            self.records = [];
            self.first_page_callback(self.indexes.length, total);
            if(indexes.length == total) {
                self.update_page(function () { self.apply_filters_callback(self.indexes.length); });
            } else {
                self.update_page(function () {
                    self.load_indexes(kwargs, null, null, function (indexes, total) {
                            self.indexes = indexes;
                            self.previous_range = {"start": 0, "end": 0};
                            self.records = [];
                            self.apply_filters_callback(self.indexes.length);
                            self.update_page();
                        }
                    );
                });
            }
        }
    );
};


ScrollPreloader.prototype.update_page = function(after_update) {
    var self = this;
    var page_range = this.get_current_page_range(self.indexes.length);

    function Range(start, end) {
        if(start <= end) {
            return {"start": Math.min(start, end), "end": Math.max(start, end)};
        } else {
            return {"start": start, "end": start};
        }
    };

    var preload_range = Range(Math.max(0, page_range.start - this.preload_margin),
                              Math.min(this.indexes.length , page_range.end + this.preload_margin));
    var range = Range(Math.max(0, page_range.start - this.chunk_size),
                      Math.min(this.indexes.length , page_range.end + this.chunk_size));
    var keep = Range(Math.max(range.start, Math.min(this.previous_range.start, range.end)),
                     Math.min(range.end, Math.max(this.previous_range.end, range.start)));
    var load_preceding = preload_range.start < this.previous_range.start
        ? Range(range.start, Math.min(this.previous_range.start, range.end))
        : Range(keep.start, keep.start);
    var load_following = preload_range.end > this.previous_range.end
        ? Range(Math.max(this.previous_range.end, range.start), range.end)
        : Range(keep.end, keep.end);


    var kept_records = this.records.slice(keep.start - this.previous_range.start,
                                          keep.end - this.previous_range.start);


    var updated = false;
    if(page_range.start >= this.previous_range.start && page_range.end <= this.previous_range.end) {
        setTimeout(function() {
            self.page_data_callback(self.records.slice(page_range.start - self.previous_range.start, page_range.end - self.previous_range.start));
        }, 0);
        updated = true;
    }

    this.load_records(self.indexes, load_preceding, function (preceding_records, preceding_range) {
        self.load_records(self.indexes, load_following, function (following_records, following_range) {
            self.records = preceding_records.concat(kept_records).concat(following_records);
            var result_range = Range(preceding_range.start, following_range.end);
            for(var record_idx = page_range.start - result_range.start;
                    record_idx < page_range.end - result_range.start;
                    ++record_idx) {
                if(typeof self.records[record_idx] === "function") {
                    self.records[record_idx] = self.records[record_idx]();
                }
            }
            if(!updated) {
                self.page_data_callback(self.records.slice(page_range.start - result_range.start, page_range.end - result_range.start));
            }
            self.previous_range = result_range;
            if(after_update) {
                after_update();
            }
        });
    });
};
