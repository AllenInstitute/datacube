var LITTLE_ENDIAN = false;

function StructuredArray(res) {
    var zbuf = atob(res.data);
    var buf = pako.inflate(zbuf);

    columns = [];
    var col_offset = 0;
    var has_unicode = false;
    for(var col_idx = 0; col_idx < res.col_names.length; ++col_idx) {
        var column = null;
        var col_start = col_offset;
        var col_length = res.item_sizes[col_idx] * res.num_rows;
        if(res.col_types[col_idx].startsWith('string') || res.col_types[col_idx].startsWith('unicode')) {
            column = buf.slice(col_offset, col_offset + col_length);
            if(res.col_types[col_idx].startsWith('unicode')) {
                has_unicode = true;
            }
        } else if(res.col_types[col_idx].startsWith('int') ||
                  res.col_types[col_idx].startsWith('uint') ||
                  res.col_types[col_idx].startsWith('float') ||
                  res.col_types[col_idx].startsWith('bool')) {
            column = new DataView(buf.buffer, col_start, col_length);
        } else {
            throw 'unsupported column type ' + res.col_types[col_idx];
        }
  
        columns.push(column);
        col_offset += res.item_sizes[col_idx] * res.num_rows;
    }
    this.columns = columns;
    this.res = res;
    if(has_unicode) {
        this.decoder = new TextDecoder('utf-8');
    }
};

StructuredArray.prototype.get_row = function(row_idx) {
    var columns = this.columns;
    var res = this.res;
    var record = {};

    for(var col_idx = 0; col_idx < res.col_names.length; ++col_idx) {
        var col_name = res.col_names[col_idx];
        var column = columns[col_idx];
        var val = null;
        if(res.col_types[col_idx].startsWith('string')) {
            var str = column.slice(row_idx * res.item_sizes[col_idx], (row_idx + 1) * res.item_sizes[col_idx]);
            var decoded = '';
            for(var i = 0; i < str.byteLength && str[i]; ++i) {
                decoded += String.fromCharCode(str[i]);
            }
            val = decoded;
        } else if(res.col_types[col_idx].startsWith('unicode')) {
            var str = column.slice(row_idx * res.item_sizes[col_idx], (row_idx + 1) * res.item_sizes[col_idx]);
            val = decoder.decode(str).replace(/\0/g, '');
        } else {
            if(res.col_types[col_idx].startsWith('int')) {
                switch(res.item_sizes[col_idx]) {
                case 1:
                    val = column.getInt8(row_idx);
                    break;
                case 2:
                    val = column.getInt16(2 * row_idx, LITTLE_ENDIAN);
                    break;
                case 4:
                    val = column.getInt32(4 * row_idx, LITTLE_ENDIAN);
                    break;
                }
            } else if(res.col_types[col_idx].startsWith('uint')) {
                switch(res.item_sizes[col_idx]) {
                case 1:
                    val = column.getUint8(row_idx);
                    break;
                case 2:
                    val = column.getUint16(2 * row_idx, LITTLE_ENDIAN);
                    break;
                case 4:
                    val = column.getUint32(4 * row_idx, LITTLE_ENDIAN);
                    break;
                }
            } else if(res.col_types[col_idx].startsWith('float')) {
                switch(res.item_sizes[col_idx]) {
                case 4:
                    val = column.getFloat32(4 * row_idx, LITTLE_ENDIAN);
                    break;
                case 8:
                    val = column.getFloat64(8 * row_idx, LITTLE_ENDIAN);
                    break;
                }
            } else if(res.col_types[col_idx].startsWith('bool')) {
                val = column.getUint8(row_idx) ? true : false;
            } else {
                throw 'unsupported column type ' + res.col_types[col_idx];
            }
        }
        record[col_name] = val;
    }
    return record;
};

StructuredArray.prototype.get_all = function() {
    var records = [];
    for(var row_idx = 0; row_idx < this.res.num_rows; ++row_idx) {
        console.log('get_row ' + row_idx + ', ' + this.res.num_rows);
        records.push(this.get_row(row_idx));
    }
    return records;
};

StructuredArray.prototype.lazy_get_all = function() {
    var records_fn = [];
    for(var row_idx = 0; row_idx < this.res.num_rows; ++row_idx) {
        let row_idx_let = row_idx;
        let sa = this;
        records_fn.push(function() {
            return sa.get_row(row_idx_let);
        });
    }
    return records_fn;
};
