/*
 *  Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
import {
    kTableSize,
    kNumChannels,
    WebRtcVad_CalcVad32khz,
    WebRtcVad_CalcVad16khz,
    WebRtcVad_CalcVad8khz,
    WebRtcVad_InitCore
} from "./vad_core.js";

var testData = require("../vad/test2.js");

const kInitCheck = 42;
const kValidRates = [8000, 16000, 32000, 48000];
const kRatesSize = 4;
const kMaxFrameLengthMs = 30;

function WebRtcVad_ValidRateAndFrameLength(rate, frame_length) {
    var return_value = -1, i, valid_length_ms, valid_length;

    // We only allow 10, 20 or 30 ms frames. Loop through valid frame rates and
    // see if we have a matching pair.
    for (i = 0; i < kRatesSize; i++) {
        if (kValidRates[i] == rate) {
            for (valid_length_ms = 10; valid_length_ms <= kMaxFrameLengthMs;
                 valid_length_ms += 10) {
                valid_length = (size_t) (kValidRates[i] / 1000 * valid_length_ms);
                if (frame_length == valid_length) {
                    return_value = 0;
                    break;
                }
            }
            break;
        }
    }

    return return_value;
}

var vad = {
    sampleRate: 16000,
    format: 'pcm',
    
    init: function(per_ms_frames = 30, mode = 2){
      per_ms_frames = Math.max(Math.min(30, per_ms_frames), 10);
      this.sampleSize = this.sampleRate * per_ms_frames / 1000;
      this.buffer = [];
      this.vadInstT = {
        vad: 0, // Speech active (=1).
        downsampling_filter_states: [0,0,0,0],
        spl_state48khzTo8khz:{
          S_48_24: [0,0,0,0,0,0,0,0],
          S_24_24: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          S_24_16: [0,0,0,0,0,0,0,0],
          S_16_8: [0,0,0,0,0,0,0,0],
        },
        noise_means: new Int16Array(kTableSize),
        speech_means: new Int16Array(kTableSize),
        noise_stds: new Int16Array(kTableSize),
        speech_stds: new Int16Array(kTableSize),
        // TODO(bjornv): Change to |frame_count|.
        frame_counter: new Int16Array(1),
        over_hang: new Int16Array(1),  // Over Hang
        num_of_speech: new Int16Array(1),
        // TODO(bjornv): Change to |age_vector|.
        index_vector: new Int16Array(16*kNumChannels),
        low_value_vector: new Int16Array(16*kNumChannels),
        // TODO(bjornv): Change to |median|.
        mean_value: new Int16Array(kNumChannels),
        upper_state: new Int16Array(5),
        lower_state: new Int16Array(5),
        hp_filter_state: new Int16Array(4),
        over_hang_max_1: undefined,
        over_hang_max_2: undefined,
        individual: undefined,
        total: undefined,
    
        init_flag: 0,
      };
      WebRtcVad_InitCore(this.vadInstT, mode);
      this.elements = new Int16Array(this.sampleSize);
      this.index = 0;
      //this.testPCMData();
    },
    testPCMData: function(){
      var len = testData.length;
      var buf = new Int16Array(len);
      for(var i = 0; i < len; i++){
        buf[i] = testData[i];
      }
      this.buffer = buf;
      this.process();
    }, 
    push: function(frameBuffer){
      //this.buffer.push(new Int16Array(frameBuffer))
      //console.log(this.buffer);
      //this.buffer.shift();
      this.buffer = new Int16Array(frameBuffer);
      this.index = 0;
      this.process();
    },
    
    process: function(){
      var self = this.vadInstT;
      var start = this.index*this.sampleSize; 
      var len = start + this.sampleSize;
      var vadArr = [];
      while(this.buffer.length > len) {
        for(var i = 0; i < this.sampleSize; i++){
          this.elements[i] = this.buffer[start+i];
        }
        this.index++;
        start = this.index*this.sampleSize;
        len = start + this.sampleSize;
        var vad;
        if (this.sampleRate == 48000) {
            //vad = WebRtcVad_CalcVad48khz(self, audio_frame, frame_length);
        } else if (this.sampleRate == 32000) {
            vad = WebRtcVad_CalcVad32khz(self, this.elements);
        } else if (this.sampleRate == 16000) {
            vad = WebRtcVad_CalcVad16khz(self, this.elements);
        } else if (this.sampleRate == 8000) {
            vad = WebRtcVad_CalcVad8khz(self, this.elements, this.elements.length);
        }
    
        if (vad > 0) {
            vad = 1;
        }
        console.log(vad);
        vadArr.push(vad);
        //return vad;
      }
      var vadCount = vadArr.reduce((total, num) => { return total + num;});
      console.log("process", vadCount, ",", vadCount/vadArr.length);
    },

  }
  
  module.exports = vad;
