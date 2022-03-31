/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
var signalProcess = require("./signal_processing.js");
// Allpass filter coefficients, upper and lower, in Q13.
// Upper: 0.64, Lower: 0.17.
const kAllPassCoefsQ13 = [5243, 1392];  // Q13.
const kSmoothingDown = 6553;  // 0.2 in Q15.
const kSmoothingUp = 32439;  // 0.99 in Q15.

// TODO(bjornv): Move this function to vad_filterbank.c.
// Downsampling filter based on splitting filter and allpass functions.
function WebRtcVad_Downsampling(signal_in, signal_out, filter_state, half_length) {
    
    var tmp16_1 = new Int16Array(1), tmp16_2 = new Int16Array(1);
    var tmp32_1 = filter_state[0];
    var tmp32_2 = filter_state[1];
    var n = 0, i = 0;
    // Downsampling by 2 gives half length.
    //var half_length = (in_length >> 1);

    // Filter coefficients in Q13, filter state in Q0.
    for (n = 0; n < half_length; n++) {
      // All-pass filtering upper branch.
      tmp16_1[0] = (tmp32_1 >> 1) + ((kAllPassCoefsQ13[0] * signal_in[i]) >> 14);
      signal_out[n] = tmp16_1[0];
      tmp32_1 = (signal_in[i]) - ((kAllPassCoefsQ13[0] * signal_out[n]) >> 12);
      i++;
      // All-pass filtering lower branch.
      tmp16_2[0] = (tmp32_2 >> 1) + ((kAllPassCoefsQ13[1] * signal_in[i]) >> 14);
      signal_out[n] += tmp16_2[0];
      tmp32_2 = (signal_in[i]) - ((kAllPassCoefsQ13[1] * tmp16_2[0]) >> 12);
      i++;
    }
    // Store the filter states.
    filter_state[0] = tmp32_1;
    filter_state[1] = tmp32_2;
}

// Inserts |feature_value| into |low_value_vector|, if it is one of the 16
// smallest values the last 100 frames. Then calculates and returns the median
// of the five smallest values.
function WebRtcVad_FindMinimum(self, feature_value, channel) {
    var i = 0, j = 0;
    var position = -1;
    // Offset to beginning of the 16 minimum values in memory.
    const offset = (channel << 4);
    var current_median = new Int16Array(1); //1600;
    var alpha = new Int16Array(1);
    var tmp32 = 0;
    current_median[0] = 1600;
    alpha[0] = 0;
    // Pointer to memory for the 16 minimum values and the age of each value of
    // the |channel|.
    //var age = self.index_vector[offset];
    var age = self.index_vector;
    //var age = self.index_vector.splice(offset);
    //var smallest_values = self.low_value_vector.splice(offset);
    var smallest_values = self.low_value_vector;
    
    // Each value in |smallest_values| is getting 1 loop older. Update |age|, and
    // remove old values.
    for (i = 0+offset; i < 16+offset; i++) {
        if (age[i] != 100) {
            age[i]++;
        } else {
            // Too old value. Remove from memory and shift larger values downwards.
            for (j = i; j < 15+offset; j++) {
                smallest_values[j] = smallest_values[j + 1];
                age[j] = age[j + 1];
            }
            age[15+offset] = 101;
            smallest_values[15+offset] = 10000;
        }
    }

    // Check if |feature_value| is smaller than any of the values in
    // |smallest_values|. If so, find the |position| where to insert the new value
    // (|feature_value|).
    if (feature_value < smallest_values[7+offset]) {
        if (feature_value < smallest_values[3+offset]) {
            if (feature_value < smallest_values[1+offset]) {
                if (feature_value < smallest_values[0+offset]) {
                    position = 0;
                } else {
                    position = 1;
                }
            } else if (feature_value < smallest_values[2+offset]) {
                position = 2;
            } else {
                position = 3;
            }
        } else if (feature_value < smallest_values[5+offset]) {
            if (feature_value < smallest_values[4+offset]) {
                position = 4;
            } else {
                position = 5;
            }
        } else if (feature_value < smallest_values[6+offset]) {
            position = 6;
        } else {
            position = 7;
        }
    } else if (feature_value < smallest_values[15+offset]) {
        if (feature_value < smallest_values[11+offset]) {
            if (feature_value < smallest_values[9+offset]) {
                if (feature_value < smallest_values[8+offset]) {
                    position = 8;
                } else {
                    position = 9;
                }
            } else if (feature_value < smallest_values[10+offset]) {
                position = 10;
            } else {
                position = 11;
            }
        } else if (feature_value < smallest_values[13+offset]) {
            if (feature_value < smallest_values[12+offset]) {
                position = 12;
            } else {
                position = 13;
            }
        } else if (feature_value < smallest_values[14+offset]) {
            position = 14;
        } else {
            position = 15;
        }
    }

    // If we have detected a new small value, insert it at the correct position
    // and shift larger values up.
    if (position > -1) {
        for (i = 15+offset; i > position+offset; i--) {
            smallest_values[i] = smallest_values[i - 1];
            age[i] = age[i - 1];
        }
        smallest_values[position+offset] = feature_value;
        age[position+offset] = 1;
    }

    // Get |current_median|.
    if (self.frame_counter > 2) {
        current_median[0] = smallest_values[2+offset];
    } else if (self.frame_counter > 0) {
        current_median[0] = smallest_values[0+offset];
    }

    // Smooth the median value.
    if (self.frame_counter > 0) {
        if (current_median[0] < self.mean_value[channel]) {
            alpha[0] = kSmoothingDown;  // 0.2 in Q15.
        } else {
            alpha[0] = kSmoothingUp;  // 0.99 in Q15.
        }
    }
    tmp32 = (alpha[0] + 1) * self.mean_value[channel];
    tmp32 += (signalProcess.WEBRTC_SPL_WORD16_MAX - alpha[0]) * current_median[0];
    tmp32 += 16384;
    self.mean_value[channel] = (tmp32 >> 15);

    //self.index_vector = self.index_vector.concat(age);
    //self.low_value_vector = self.low_value_vector.concat(smallest_values);

    return self.mean_value[channel];
}

module.exports = {
    WebRtcVad_FindMinimum,
    WebRtcVad_Downsampling
}
