'use strict';

import TextDecoder from "text-encoding";

/**
 * sa.data
 * sa.col_names
 * sa.col_types
 * sa.item_sizes
 * sa.num_rows
 */
export function StructuredArray(sa, little_endian) {
    this.little_endian = typeof(little_endian) === 'undefined' ? false : little_endian;

    var buf = sa.data;

    var columns = [];
    var col_offset = 0;
    var has_unicode = false;
    for(var col_idx = 0; col_idx < sa.col_names.length; ++col_idx) {
        var column = null;
        var col_start = col_offset;
        var col_length = sa.item_sizes[col_idx] * sa.num_rows;
        if(sa.col_types[col_idx].startsWith('string') || sa.col_types[col_idx].startsWith('unicode')) {
            column = buf.slice(col_offset, col_offset + col_length);
            if(sa.col_types[col_idx].startsWith('unicode')) {
                has_unicode = true;
            }
        } else if(sa.col_types[col_idx].startsWith('int') ||
                  sa.col_types[col_idx].startsWith('uint') ||
                  sa.col_types[col_idx].startsWith('float') ||
                  sa.col_types[col_idx].startsWith('bool')) {
            column = new DataView(buf.buffer, col_start, col_length);
        } else {
            throw 'unsupported column type ' + sa.col_types[col_idx];
        }
  
        columns.push(column);
        col_offset += sa.item_sizes[col_idx] * sa.num_rows;
    }
    this.columns = columns;
    this.sa = sa;
    if(has_unicode) {
        this.decoder = new TextDecoder('utf-8');
    }
};

StructuredArray.prototype.get_row = function(row_idx) {
    var columns = this.columns;
    var sa = this.sa;
    var record = {};

    for(var col_idx = 0; col_idx < sa.col_names.length; ++col_idx) {
        var col_name = sa.col_names[col_idx];
        var column = columns[col_idx];
        var val = null;
        if(sa.col_types[col_idx].startsWith('string')) {
            var str = column.slice(row_idx * sa.item_sizes[col_idx], (row_idx + 1) * sa.item_sizes[col_idx]);
            var decoded = '';
            for(var i = 0; i < str.byteLength && str[i]; ++i) {
                decoded += String.fromCharCode(str[i]);
            }
            val = decoded;
        } else if(sa.col_types[col_idx].startsWith('unicode')) {
            var str = column.slice(row_idx * sa.item_sizes[col_idx], (row_idx + 1) * sa.item_sizes[col_idx]);
            val = decoder.decode(str).replace(/\0/g, '');
        } else {
            if(sa.col_types[col_idx].startsWith('int')) {
                switch(sa.item_sizes[col_idx]) {
                case 1:
                    val = column.getInt8(row_idx);
                    break;
                case 2:
                    val = column.getInt16(2 * row_idx, this.little_endian);
                    break;
                case 4:
                    val = column.getInt32(4 * row_idx, this.little_endian);
                    break;
                }
            } else if(sa.col_types[col_idx].startsWith('uint')) {
                switch(sa.item_sizes[col_idx]) {
                case 1:
                    val = column.getUint8(row_idx);
                    break;
                case 2:
                    val = column.getUint16(2 * row_idx, this.little_endian);
                    break;
                case 4:
                    val = column.getUint32(4 * row_idx, this.little_endian);
                    break;
                }
            } else if(sa.col_types[col_idx].startsWith('float')) {
                switch(sa.item_sizes[col_idx]) {
                case 4:
                    val = column.getFloat32(4 * row_idx, this.little_endian);
                    break;
                case 8:
                    val = column.getFloat64(8 * row_idx, this.little_endian);
                    break;
                }
            } else if(sa.col_types[col_idx].startsWith('bool')) {
                val = column.getUint8(row_idx) ? true : false;
            } else {
                throw 'unsupported column type ' + sa.col_types[col_idx];
            }
        }
        record[col_name] = val;
    }
    return record;
};

StructuredArray.prototype.get_all = function() {
    var records = [];
    for(var row_idx = 0; row_idx < this.sa.num_rows; ++row_idx) {
        records.push(this.get_row(row_idx));
    }
    return records;
};

StructuredArray.prototype.lazy_get_all = function() {
    var records_fn = [];
    for(var row_idx = 0; row_idx < this.sa.num_rows; ++row_idx) {
        let row_idx_let = row_idx;
        let self = this;
        records_fn.push(function() {
            return self.get_row(row_idx_let);
        });
    }
    return records_fn;
};
