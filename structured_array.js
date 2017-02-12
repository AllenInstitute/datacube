// performance is the same either way; just a matter of FileReader needing to be
// asynchronous but TextDecoder being an external dependency (polyfill)
var USE_ENCODING_POLYFILL = true;
var LITTLE_ENDIAN = false;

function decode_structured_array(res, callback) {
    var zbuf = atob(res.data);
    var buf = pako.inflate(zbuf);

    function build_records(decoded_cols) {
        var records = [];
        for(var row_idx = 0; row_idx < res.num_rows; ++row_idx) {
            records.push({});
        }

        var col_offset = 0;
        var string_col_idx = 0;
        for(var col_idx = 0; col_idx < res.col_names.length; ++col_idx) {
            var col_name = res.col_names[col_idx];
            var column_buf = buf.slice(col_offset, col_offset + res.item_sizes[col_idx] * res.num_rows);
            var column_arr = new DataView(column_buf.buffer);
            var is_string_col = res.col_types[col_idx].startsWith('string');

            var string_start = 0;
            for(var row_idx = 0; row_idx < res.num_rows; ++row_idx) {
                if(is_string_col) {
                    var chunk = decoded_cols[string_col_idx].substring(string_start, string_start + res.item_sizes[col_idx]);
                    var match = (chunk.match(/^[^\0]*/g) || [""])[0];
                    records[row_idx][col_name] = match;
                    
                    // steps aren't uniform since this is decoded UTF-8. regex finds the start of
                    // the next string (service guarantees at least one intervening '\0'. if the next
                    // string is empty, the step could be inaccurate. but since empty strings will
                    // always result in chunks of predictable size, only the first such step will be
                    // inaccurate if there are more subsequent empty strings. arriving at the next
                    // non-empty string, the regex will correct for the misalignment as usual.
                    var step = chunk.search(/\0[^\0]/g) + 1;
                    if(step > 0) {
                        string_start += step;
                    } else {
                        string_start += res.item_sizes[col_idx];
                    }
                } else {
                    var val = null;
                    if(res.col_types[col_idx].startsWith('int')) {
                        switch(res.item_sizes[col_idx]) {
                        case 1:
                            val = column_arr.getInt8(row_idx);
                            break;
                        case 2:
                            val = column_arr.getInt16(row_idx, LITTLE_ENDIAN);
                            break;
                        case 4:
                            val = column_arr.getInt32(row_idx, LITTLE_ENDIAN);
                            break;
                        }
                    } else if(res.col_types[col_idx].startsWith('uint')) {
                        switch(res.item_sizes[col_idx]) {
                        case 1:
                            val = column_arr.getUint8(row_idx);
                            break;
                        case 2:
                            val = column_arr.getUint16(row_idx, LITTLE_ENDIAN);
                            break;
                        case 4:
                            val = column_arr.getUint32(row_idx, LITTLE_ENDIAN);
                            break;
                        }
                    } else if(res.col_types[col_idx].startsWith('float')) {
                        switch(res.item_sizes[col_idx]) {
                        case 4:
                            val = column_arr.getFloat32(row_idx, LITTLE_ENDIAN);
                            break;
                        case 8:
                            val = column_arr.getFloat64(row_idx, LITTLE_ENDIAN);
                            break;
                        }
                    } else if(res.col_types[col_idx].startsWith('bool')) {
                        val = column_arr.getUint8(row_idx) ? true : false;
                    } else {
                        throw 'unsupported column type ' + res.col_types[col_idx];
                    }
                    records[row_idx][col_name] = val;
                }
            }
            
            if(is_string_col) {
                ++string_col_idx;
            }
            col_offset += res.item_sizes[col_idx] * res.num_rows;
        }
        return records;
    }


    if(!USE_ENCODING_POLYFILL) {
        console.log('using FileReader');
        function largeuint8ArrToString(uint8arr) {
            return new Promise(function(resolve, reject) {
                var bb = new Blob([uint8arr]);
                var f = new FileReader();
                f.onload = function(e) {
                    resolve(e.target.result);
                };
                f.onerror = function() {
                    return reject(this);
                }

                f.readAsText(bb);
            })
        }


        decode_promises = [];
        var col_offset = 0;
        for(var col_idx = 0; col_idx < res.col_names.length; ++col_idx) {
            if(res.col_types[col_idx].startsWith('string') || res.col_types[col_idx].startsWith('unicode')) {
                var column_buf = buf.slice(col_offset, col_offset + res.item_sizes[col_idx] * res.num_rows);
                decode_promises.push(largeuint8ArrToString(column_buf));
            }
            col_offset += res.item_sizes[col_idx] * res.num_rows;
        }


        Promise.all(decode_promises).then(function(decoded_cols) {
            var records = build_records(decoded_cols);
            callback(records);
        });
    } else {
        console.log('using TextDecoder');
        decoded_cols = [];
        var col_offset = 0;
        var decoder = new TextDecoder('utf-8');
        for(var col_idx = 0; col_idx < res.col_names.length; ++col_idx) {
            if(res.col_types[col_idx].startsWith('string') || res.col_types[col_idx].startsWith('unicode')) {
                var column_buf = buf.slice(col_offset, col_offset + res.item_sizes[col_idx] * res.num_rows);
                decoded_cols.push(decoder.decode(column_buf));
            }
            col_offset += res.item_sizes[col_idx] * res.num_rows;
        }

        var records = build_records(decoded_cols);
        callback(records);
    }
};
