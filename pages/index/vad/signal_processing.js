

// TODO(bugs.webrtc.org/9553): These function pointers are useless. Refactor
// things so that we simply have a bunch of regular functions with different
// implementations for different platforms.

const WEBRTC_SPL_WORD16_MAX = 32767;
const WEBRTC_SPL_WORD16_MIN = -32768;
const WEBRTC_SPL_WORD32_MAX = 0x7fffffff;
const WEBRTC_SPL_WORD32_MIN = 0x80000000;
const WEBRTC_SPL_MIN = (A, B)=>{ return A < B ? A : B}  // Get min value
// TODO(kma/bjorn): For the next two macros, investigate how to correct the code
// for inputs of a = WEBRTC_SPL_WORD16_MIN or WEBRTC_SPL_WORD32_MIN.

const WEBRTC_SPL_MUL = (a, b)=> {return ((a) * (b))}
const WEBRTC_SPL_MUL_16_U16 = (a, b)=> {return (a) *(b)}

const WEBRTC_SPL_MUL_16_16 = WEBRTC_SPL_MUL_16_U16;

// C + the 32 most significant bits of A * B
const WEBRTC_SPL_SCALEDIFF32 = (A, B, C) => {
    return (C + (B >> 16) * A + (((B & 0x0000FFFF) * A) >> 16));
}
/*
const MaxAbsValueW16 WebRtcSpl_MaxAbsValueW16 = WebRtcSpl_MaxAbsValueW16C;
const MaxAbsValueW32 WebRtcSpl_MaxAbsValueW32 = WebRtcSpl_MaxAbsValueW32C;
const MaxValueW16 WebRtcSpl_MaxValueW16 = WebRtcSpl_MaxValueW16C;
const MaxValueW32 WebRtcSpl_MaxValueW32 = WebRtcSpl_MaxValueW32C;
const MinValueW16 WebRtcSpl_MinValueW16 = WebRtcSpl_MinValueW16C;
const MinValueW32 WebRtcSpl_MinValueW32 = WebRtcSpl_MinValueW32C;
*/

// Table used by WebRtcSpl_CountLeadingZeros32_NotBuiltin. For each uint32_t n
// that's a sequence of 0 bits followed by a sequence of 1 bits, the entry at
// index (n * 0x8c0b2891) >> 26 in this table gives the number of zero bits in
// n.
const kWebRtcSpl_CountLeadingZeros32_Table = [
        32, 8, 17, -1, -1, 14, -1, -1, -1, 20, -1, -1, -1, 28, -1, 18,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 26, 25, 24,
        4, 11, 23, 31, 3, 7, 10, 16, 22, 30, -1, -1, 2, 6, 13, 9,
        -1, 15, -1, 21, -1, 29, 19, -1, -1, -1, -1, -1, 1, 27, 5, 12,
];

function spl_countLeadingZeros32_notBuiltin(n) {
    // Normalize n by rounding up to the nearest number that is a sequence of 0
    // bits followed by a sequence of 1 bits. This number has the same number of
    // leading zeros as the original n. There are exactly 33 such values.
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;

    // Multiply the modified n with a constant selected (by exhaustive search)
    // such that each of the 33 possible values of n give a product whose 6 most
    // significant bits are unique. Then look up the answer in the table.
    var i = (n * 0x8c0b2891) >> 26;
    i = i > 0? i : 64+i;
    return kWebRtcSpl_CountLeadingZeros32_Table[i];
    //return kWebRtcSpl_CountLeadingZeros32_Table[(n * 0x8c0b2891) >> 26];
}
function WebRtcSpl_NormU32(a) {
    return a == 0 ? 0 : spl_countLeadingZeros32_notBuiltin(a);
}
function WebRtcSpl_NormW32(a) {
    return a == 0 ? 0 : spl_countLeadingZeros32_notBuiltin(a < 0 ? ~a : a) - 1;
}
function WebRtcSpl_GetSizeInBits(n) {
    return 32 - spl_countLeadingZeros32_notBuiltin(n);
}

function WebRtcSpl_DivW32W16(num, den) {
    // Guard against division with 0
    if (den != 0) {
        return parseInt(num / den);
    } else {
        return 0x7FFFFFFF;
    }
}



// TODO(bjorn/kma): Consolidate function pairs (e.g. combine
//   WebRtcSpl_MaxAbsValueW16C and WebRtcSpl_MaxAbsIndexW16 into a single one.)
// TODO(kma): Move the next six functions into min_max_operations_c.c.

// Maximum absolute value of word16 vector. C version for generic platforms.
function WebRtcSpl_MaxAbsValueW16C(vector, length) {
    var i = 0;
    var absolute = 0, maximum = 0;

    for (i = 0; i < length; i++) {
        absolute = Math.abs(parseInt(vector[i]));

        if (absolute > maximum) {
            maximum = absolute;
        }
    }

    // Guard the case for abs(-32768).
    if (maximum > WEBRTC_SPL_WORD16_MAX) {
        maximum = WEBRTC_SPL_WORD16_MAX;
    }

    return maximum;
}

// Maximum absolute value of word32 vector. C version for generic platforms.
function WebRtcSpl_MaxAbsValueW32C(vector, length) {
    // Use uint32_t for the local variables, to accommodate the return value
    // of abs(0x80000000), which is 0x80000000.

    var absolute = 0, maximum = 0;
    var i = 0;

    for (i = 0; i < length; i++) {
        absolute = Math.abs(parseInt(vector[i]));
        if (absolute > maximum) {
            maximum = absolute;
        }
    }

    maximum = WEBRTC_SPL_MIN(maximum, WEBRTC_SPL_WORD32_MAX);

    return maximum;
}

// Maximum value of word16 vector. C version for generic platforms.
function WebRtcSpl_MaxValueW16C(vector, length) {
    var maximum = WEBRTC_SPL_WORD16_MIN;
    var i = 0;


    for (i = 0; i < length; i++) {
        if (vector[i] > maximum)
            maximum = vector[i];
    }
    return maximum;
}

// Maximum value of word32 vector. C version for generic platforms.
function WebRtcSpl_MaxValueW32C(vector, length) {
    var maximum = WEBRTC_SPL_WORD32_MIN;
    var i = 0;

    RTC_DCHECK_GT(length, 0);

    for (i = 0; i < length; i++) {
        if (vector[i] > maximum)
            maximum = vector[i];
    }
    return maximum;
}

// Minimum value of word16 vector. C version for generic platforms.
function WebRtcSpl_MinValueW16C(vector, length) {
    var minimum = WEBRTC_SPL_WORD16_MAX;
    var i = 0;

    for (i = 0; i < length; i++) {
        if (vector[i] < minimum)
            minimum = vector[i];
    }
    return minimum;
}

// Minimum value of word32 vector. C version for generic platforms.
function WebRtcSpl_MinValueW32C(vector, length) {
    var minimum = WEBRTC_SPL_WORD32_MAX;
    var i = 0;


    for (i = 0; i < length; i++) {
        if (vector[i] < minimum)
            minimum = vector[i];
    }
    return minimum;
}

// allpass filter coefficients.
const kResampleAllpass = [
    [821,  6110, 12382],
    [3050, 9368, 15063]
];

//
//   decimator
// input:  int32_t (shifted 15 positions to the left, + offset 16384) OVERWRITTEN!
// output: int16_t (saturated) (of length len/2)
// state:  filter state array; length = 8

function WebRtcSpl_DownBy2IntToShort(_in, len, out, state) {
    var tmp0, tmp1, diff;
    var i;

    len >>= 1;

// lower allpass filter (operates on even input samples)
    for (i = 0; i < len; i++) {
        tmp0 = _in[i << 1];
        diff = tmp0 - state[1];
// UBSan: -1771017321 - 999586185 cannot be represented in type 'int'

// scale down and round
        diff = (diff + (1 << 13)) >> 14;
        tmp1 = state[0] + diff * kResampleAllpass[1][0];
        state[0] = tmp0;
        diff = tmp1 - state[2];
// scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        tmp0 = state[1] + diff * kResampleAllpass[1][1];
        state[1] = tmp1;
        diff = tmp0 - state[3];
// scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        state[3] = state[2] + diff * kResampleAllpass[1][2];
        state[2] = tmp0;

// divide by two and store temporarily
        _in[i << 1] = (state[3] >> 1);
    }

    //in++;
    var index = 1;
// upper allpass filter (operates on odd input samples)
    for (i = 0; i < len; i++) {
        tmp0 = _in[index];
        index += 2;
        diff = tmp0 - state[5];
// scale down and round
        diff = (diff + (1 << 13)) >> 14;
        tmp1 = state[4] + diff * kResampleAllpass[0][0];
        state[4] = tmp0;
        diff = tmp1 - state[6];
// scale down and round
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        tmp0 = state[5] + diff * kResampleAllpass[0][1];
        state[5] = tmp1;
        diff = tmp0 - state[7];
// scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        state[7] = state[6] + diff * kResampleAllpass[0][2];
        state[6] = tmp0;

// divide by two and store temporarily
        _in[i << 1] = (state[7] >> 1);
    }

    //in--;

// combine allpass outputs
    for (i = 0; i < len; i += 2) {
// divide by two, add both allpass outputs and round
        tmp0 = (_in[i << 1] + _in[(i << 1) + 1]) >> 15;
        tmp1 = (_in[(i << 1) + 2] + _in[(i << 1) + 3]) >> 15;
        if (tmp0 > 0x00007FFF)
            tmp0 = 0x00007FFF;
        if (tmp0 < 0xFFFF8000)
            tmp0 = 0xFFFF8000;
        out[i] = tmp0;
        if (tmp1 >  0x00007FFF)
            tmp1 = 0x00007FFF;
        if (tmp1 <  0xFFFF8000)
            tmp1 = 0xFFFF8000;
        out[i + 1] =  tmp1;
    }
}

//
//   decimator
// input:  int16_t
// output: int32_t (shifted 15 positions to the left, + offset 16384) (of length len/2)
// state:  filter state array; length = 8

function WebRtcSpl_DownBy2ShortToInt(_in, len, out, state) {
    var tmp0, tmp1, diff;
    var i;

    len >>= 1;

    // lower allpass filter (operates on even input samples)
    for (i = 0; i < len; i++) {
        tmp0 = (_in[i << 1] << 15) + (1 << 14);
        diff = tmp0 - state[1];
        // scale down and round
        diff = (diff + (1 << 13)) >> 14;
        tmp1 = state[0] + diff * kResampleAllpass[1][0];
        state[0] = tmp0;
        diff = tmp1 - state[2];
        // UBSan: -1379909682 - 834099714 cannot be represented in type 'int'

        // scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        tmp0 = state[1] + diff * kResampleAllpass[1][1];
        state[1] = tmp1;
        diff = tmp0 - state[3];
        // scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        state[3] = state[2] + diff * kResampleAllpass[1][2];
        state[2] = tmp0;

        // divide by two and store temporarily
        out[i] = (state[3] >> 1);
    }

    var index = 1;

    // upper allpass filter (operates on odd input samples)
    for (i = 0; i < len; i++) {
        tmp0 = (_in[index] << 15) + (1 << 14);
        index += 2;
        diff = tmp0 - state[5];
        // scale down and round
        diff = (diff + (1 << 13)) >> 14;
        tmp1 = state[4] + diff * kResampleAllpass[0][0];
        state[4] = tmp0;
        diff = tmp1 - state[6];
        // scale down and round
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        tmp0 = state[5] + diff * kResampleAllpass[0][1];
        state[5] = tmp1;
        diff = tmp0 - state[7];
        // scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        state[7] = state[6] + diff * kResampleAllpass[0][2];
        state[6] = tmp0;

        // divide by two and store temporarily
        out[i] += (state[7] >> 1);
    }
}

//   lowpass filter
// input:  int32_t (shifted 15 positions to the left, + offset 16384)
// output: int32_t (normalized, not saturated)
// state:  filter state array; length = 8
function WebRtcSpl_LPBy2IntToInt(_in, len, out, state) {
    var tmp0, tmp1, diff;
    var i;

    len >>= 1;

    // lower allpass filter: odd input -> even output samples
    //in++;
    var index = 1;
    // initial state of polyphase delay element
    tmp0 = state[12];
    for (i = 0; i < len; i++) {
        diff = tmp0 - state[1];
        // scale down and round
        diff = (diff + (1 << 13)) >> 14;
        tmp1 = state[0] + diff * kResampleAllpass[1][0];
        state[0] = tmp0;
        diff = tmp1 - state[2];
        // scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        tmp0 = state[1] + diff * kResampleAllpass[1][1];
        state[1] = tmp1;
        diff = tmp0 - state[3];
        // scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        state[3] = state[2] + diff * kResampleAllpass[1][2];
        state[2] = tmp0;

        // scale down, round and store
        out[i << 1] = state[3] >> 1;
        tmp0 = _in[index];
        index += 2;
    }
    //in--;

    // upper allpass filter: even input -> even output samples
    for (i = 0; i < len; i++) {
        tmp0 = _in[i << 1];
        diff = tmp0 - state[5];
        // UBSan: -794814117 - 1566149201 cannot be represented in type 'int'

        // scale down and round
        diff = (diff + (1 << 13)) >> 14;
        tmp1 = state[4] + diff * kResampleAllpass[0][0];
        state[4] = tmp0;
        diff = tmp1 - state[6];
        // scale down and round
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        tmp0 = state[5] + diff * kResampleAllpass[0][1];
        state[5] = tmp1;
        diff = tmp0 - state[7];
        // scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        state[7] = state[6] + diff * kResampleAllpass[0][2];
        state[6] = tmp0;

        // average the two allpass outputs, scale down and store
        out[i << 1] = (out[i << 1] + (state[7] >> 1)) >> 15;
    }

    // switch to odd output samples
    //out++;
    index = 1;
    // lower allpass filter: even input -> odd output samples
    for (i = 0; i < len; i++) {
        tmp0 = _in[i << 1];
        diff = tmp0 - state[9];
        // scale down and round
        diff = (diff + (1 << 13)) >> 14;
        tmp1 = state[8] + diff * kResampleAllpass[1][0];
        state[8] = tmp0;
        diff = tmp1 - state[10];
        // scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        tmp0 = state[9] + diff * kResampleAllpass[1][1];
        state[9] = tmp1;
        diff = tmp0 - state[11];
        // scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        state[11] = state[10] + diff * kResampleAllpass[1][2];
        state[10] = tmp0;

        // scale down, round and store
        out[index] = state[11] >> 1;
        index += 2;
    }

    // upper allpass filter: odd input -> odd output samples
    //in++;
    index = 1;
    for (i = 0; i < len; i++) {
        tmp0 = _in[index];
        index += 2;
        diff = tmp0 - state[13];
        // scale down and round
        diff = (diff + (1 << 13)) >> 14;
        tmp1 = state[12] + diff * kResampleAllpass[0][0];
        state[12] = tmp0;
        diff = tmp1 - state[14];
        // scale down and round
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        tmp0 = state[13] + diff * kResampleAllpass[0][1];
        state[13] = tmp1;
        diff = tmp0 - state[15];
        // scale down and truncate
        diff = diff >> 14;
        if (diff < 0)
            diff += 1;
        state[15] = state[14] + diff * kResampleAllpass[0][2];
        state[14] = tmp0;

        // average the two allpass outputs, scale down and store
        out[i << 1] = (out[i << 1] + (state[15] >> 1)) >> 15;
    }
}

// interpolation coefficients
const kCoefficients48To32 = [
    [778, -2050, 1087,  23285, 12903, -3783, 441,   222],
    [222, 441,   -3783, 12903, 23285, 1087,  -2050, 778]
];


//   Resampling ratio: 2/3
// input:  int32_t (normalized, not saturated) :: size 3 * K
// output: int32_t (shifted 15 positions to the left, + offset 16384) :: size 2 * K
//      K: number of blocks

function WebRtcSpl_Resample48khzTo32khz(_In, Out, K) {
    /////////////////////////////////////////////////////////////
    // Filter operation:
    //
    // Perform resampling (3 input samples -> 2 output samples);
    // process in sub blocks of size 3 samples.
    var tmp;
    var m, inIndex = 0, outIndex = 0;

    for (m = 0; m < K; m++) {
        tmp = 1 << 14;
        tmp += kCoefficients48To32[0][0] * _In[inIndex+0];
        tmp += kCoefficients48To32[0][1] * _In[inIndex+1];
        tmp += kCoefficients48To32[0][2] * _In[inIndex+2];
        tmp += kCoefficients48To32[0][3] * _In[inIndex+3];
        tmp += kCoefficients48To32[0][4] * _In[inIndex+4];
        tmp += kCoefficients48To32[0][5] * _In[inIndex+5];
        tmp += kCoefficients48To32[0][6] * _In[inIndex+6];
        tmp += kCoefficients48To32[0][7] * _In[inIndex+7];
        Out[outIndex+0] = tmp;

        tmp = 1 << 14;
        tmp += kCoefficients48To32[1][0] * _In[inIndex+1];
        tmp += kCoefficients48To32[1][1] * _In[inIndex+2];
        tmp += kCoefficients48To32[1][2] * _In[inIndex+3];
        tmp += kCoefficients48To32[1][3] * _In[inIndex+4];
        tmp += kCoefficients48To32[1][4] * _In[inIndex+5];
        tmp += kCoefficients48To32[1][5] * _In[inIndex+6];
        tmp += kCoefficients48To32[1][6] * _In[inIndex+7];
        tmp += kCoefficients48To32[1][7] * _In[inIndex+8];
        Out[outIndex+1] = tmp;

        // update pointers
        //_In += 3;
        inIndex += 3;
        //Out += 2;
        outIndex += 2;
    }
}


// allpass filter coefficients.
const kResampleAllpass1 = [3284, 24441, 49528 << 15];
const kResampleAllpass2 = [12199, 37471 << 15, 60255 << 15];

// Multiply two 32-bit values and accumulate to another input value.
// Return: Return: state + (((diff << 1) * tbl_value) >> 32)
//
// The reason to introduce this function is that, in case we can't use smlawb
// instruction (in MUL_ACCUM_1) due to input value range, we can still use
// smmla to save some cycles.

// Multiply a 32-bit value with a 16-bit value and accumulate to another input:
const MUL_ACCUM_1 = WEBRTC_SPL_SCALEDIFF32;
const MUL_ACCUM_2 = WEBRTC_SPL_SCALEDIFF32;



////////////////////////////
///// 48 kHz ->  8 kHz /////
////////////////////////////

// 48 -> 8 resampler
function WebRtcSpl_Resample48khzTo8khz(_in, out, state, tmpmem) {
    ///// 48 --> 24 /////
    // int16_t  in[480]
    // int32_t out[240]
    /////
    WebRtcSpl_DownBy2ShortToInt(_in, 480, tmpmem + 256, state.S_48_24);

    ///// 24 --> 24(LP) /////
    // int32_t  in[240]
    // int32_t out[240]
    /////
    WebRtcSpl_LPBy2IntToInt(tmpmem + 256, 240, tmpmem + 16, state.S_24_24);

    ///// 24 --> 16 /////
    // int32_t  in[240]
    // int32_t out[160]
    /////
    // copy state to and from input array
    //memcpy(tmpmem + 8, state->S_24_16, 8 * sizeof(int32_t));
    //memcpy(state->S_24_16, tmpmem + 248, 8 * sizeof(int32_t));
    WebRtcSpl_Resample48khzTo32khz(tmpmem + 8, tmpmem, 80);

    ///// 16 --> 8 /////
    // int32_t  in[160]
    // int16_t out[80]
    /////
    WebRtcSpl_DownBy2IntToShort(tmpmem, 160, out, state.S_16_8);
}


////////////////////////////
/////  8 kHz -> 48 kHz /////
////////////////////////////

function WebRtcSpl_GetScalingSquare(in_vector, in_vector_length, times) {
    var nbits = WebRtcSpl_GetSizeInBits(times),
    smax = -1, sabs, t;

    for(var i = 0; i < in_vector_length; i++){
        sabs = Math.abs(in_vector[i]);
        smax = (sabs > smax ? sabs : smax);
    }
    t = WebRtcSpl_NormW32(WEBRTC_SPL_MUL(smax, smax));

    if (smax == 0) {
        return 0; // Since norm(0) returns 0
    } else {
        return (t > nbits) ? 0 : nbits - t;
    }
}

function WebRtcSpl_Energy(vector, vector_length) {
    var en = 0;
    var i;
    var scaling = WebRtcSpl_GetScalingSquare(vector, vector_length, vector_length);

    for (i = 0; i < vector_length; i++) {
        en += (vector[i] * vector[i]) >> scaling;
    }
    //scale_factor = scaling;

    return {energy: en, scale_factor: scaling};
}

module.exports = {
    WEBRTC_SPL_WORD16_MAX,
    WebRtcSpl_DivW32W16,
    WebRtcSpl_NormU32,
    WebRtcSpl_NormW32,
    WebRtcSpl_Energy
}
