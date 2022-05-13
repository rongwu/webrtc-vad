var signalProcess = require("./signal_processing.js");

// Constants used in LogOfEnergy().
const kLogConst = 24660;  // 160*log10(2) in Q9.
const kLogEnergyIntPart = 14336;  // 14 in Q10
const kMinEnergy = 10;

// Coefficients used by HighPassFilter, Q14.
const kHpZeroCoefs = [6631, -13262, 6631];
const kHpPoleCoefs = [16384, -7756, 5620];

// Allpass filter coefficients, upper and lower, in Q15.
// Upper: 0.64, Lower: 0.17
const kAllPassCoefsQ15 = [20972, 5571];

// Adjustment for division with two in SplitFilter.
const kOffsetVector = [368, 368, 272, 176, 176, 176];

// High pass filtering, with a cut-off frequency at 80 Hz, if the |data_in| is
// sampled at 500 Hz.
//
// - data_in      [i]   : Input audio data sampled at 500 Hz.
// - data_length  [i]   : Length of input and output data.
// - filter_state [i/o] : State of the filter.
// - data_out     [o]   : Output audio data in the frequency interval
//                        80 - 250 Hz.
function highPassFilter(data_in, data_length, filter_state, data_out) {
    var i;
    var in_ptr = data_in;
    var out_ptr = data_out;
    var tmp32 = 0;


    // The sum of the absolute values of the impulse response:
    // The zero/pole-filter has a max amplification of a single sample of: 1.4546
    // Impulse response: 0.4047 -0.6179 -0.0266  0.1993  0.1035  -0.0194
    // The all-zero section has a max amplification of a single sample of: 1.6189
    // Impulse response: 0.4047 -0.8094  0.4047  0       0        0
    // The all-pole section has a max amplification of a single sample of: 1.9931
    // Impulse response: 1.0000  0.4734 -0.1189 -0.2187 -0.0627   0.04532

    for (i = 0; i < data_length; i++) {
        // All-zero section (filter coefficients in Q14).
        tmp32 = kHpZeroCoefs[0] * in_ptr[i];
        tmp32 += kHpZeroCoefs[1] * filter_state[0];
        tmp32 += kHpZeroCoefs[2] * filter_state[1];
        filter_state[1] = filter_state[0];
        filter_state[0] = in_ptr[i];

        // All-pole section (filter coefficients in Q14).
        tmp32 -= kHpPoleCoefs[1] * filter_state[2];
        tmp32 -= kHpPoleCoefs[2] * filter_state[3];
        filter_state[3] = filter_state[2];
        filter_state[2] = (tmp32 >> 14);
        out_ptr[i] = filter_state[2];
    }
}

// All pass filtering of |data_in|, used before splitting the signal into two
// frequency bands (low pass vs high pass).
// Note that |data_in| and |data_out| can NOT correspond to the same address.
//
// - data_in            [i]   : Input audio signal given in Q0.
// - data_length        [i]   : Length of input and output data.
// - filter_coefficient [i]   : Given in Q15.
// - filter_state       [i/o] : State of the filter given in Q(-1).
// - data_out           [o]   : Output audio signal given in Q(-1).
function allPassFilter(data_in, dIndex, data_length, filter_coefficient, filter_state,  frequency_band, data_out) {
    // The filter can only cause overflow (in the w16 output variable)
    // if more than 4 consecutive input numbers are of maximum value and
    // has the the same sign as the impulse responses first taps.
    // First 6 taps of the impulse response:
    // 0.6399 0.5905 -0.3779 0.2418 -0.1547 0.0990
    /*
    数字滤波器的差分方程表示为：
    令x(n)为输入序列， y(n)为输出序列， c为实系数filter_coefficient ，可知表示 :
    y(n)=x(n−1)−c∗y(n−1)+c∗x(n)
    但由于函数中data_in +=2, 即反馈是只与历史第二值相关的，其表示为：
    y(n)=x(n−2)−c∗y(n−2)+c∗x(n)
    则传输函数为：
    H(z) = (c+z^−2)/(1+c*z^−2)​
    是一个2阶全通滤波器，∣H(w)∣=1。
    滤波器当前的输出仅依赖于输入，而不依赖过去的输出，称为非递归滤波器(FIR)
    反之，滤波器当前的输出依赖于输入和过去的输出，称为递归滤波器（IIR)，N为递归滤波器的阶数。
    一对互补的低通和高通传输函数可有两个稳定的全通滤波器并联组成
    F(z)=1/2​(A0​(z)+A1​(z))
    G(z)=1/2(A0​(z)−A1​(z))
    第一个全通滤波器的参数kAllPassCoefsQ15[0] = 20972, 即 c = 20972/(2^15) = 0.64，则传输函数：
    A0​(z) = (0.64+z^−2)/(1+0.64*z^−2)​
    第二个全通滤波器的参数kAllPassCoefsQ15[0] = 5571, 即 c = 5571/(2^15) = 0.17， 则传输函数：
    A1(z) = (0.17+z^−2)/(1+0.17*z^−2)​
    第一个全通滤波器的输入序列data_in[n]是第二个全通函数输入序列data_in[n+1]的移位，且全通滤波器中输出是 tmp16 = (int16_t) (tmp32 >> 16); // Q(-1) *data_out++ = tmp16;Q(-1), 即 1/2的输出，则传输函数为
    H(z) = 1/2(A1(z) + z^-1*A0(z)) 和 H(z) = 1/2(A1(z) - z^-1*A0(z))
    函数中*hp_data_out++ -= *lp_data_out;对应的高通传输函数：
    H(z) = 1/2(A1(z) + z^-1*A0(z))
         = 1/2((0.17+z^−2)/(1+0.17*z^−2)​ + z^-1*(0.64+z^−2)/(1+0.64*z^−2)​)
    函数中*lp_data_out++ += tmp_out;对应的低通传输函数：
    H(z) = 1/2(A1(z) - z^-1*A0(z))
         = 1/2((0.17+z^−2)/(1+0.17*z^−2)​ - z^-1*(0.64+z^−2)/(1+0.64*z^−2)​)
    */
    var i;
    var tmp16 = new Int16Array(new ArrayBuffer(2));
    tmp16[0] = 0;
    var tmp32 = 0;
    var state32 = ((filter_state[frequency_band]) * (1 << 16));  // Q15

    for (i = 0; i < data_length; i++) {
        tmp32 = state32 + filter_coefficient * data_in[dIndex];
        tmp16[0] = (tmp32 >> 16);  // Q(-1)
        data_out[i] = tmp16[0];
        state32 = (data_in[dIndex] * (1 << 14)) - filter_coefficient * tmp16[0];  // Q14
        state32 *= 2;  // Q15.
        dIndex += 2;
        //console.log(tmp32, tmp16, state32)
    }

    filter_state[frequency_band] = (state32 >> 16);  // Q(-1)
}

// Splits |data_in| into |hp_data_out| and |lp_data_out| corresponding to
// an upper (high pass) part and a lower (low pass) part respectively.
//
// - data_in      [i]   : Input audio data to be split into two frequency bands.
// - data_length  [i]   : Length of |data_in|.
// - upper_state  [i/o] : State of the upper filter, given in Q(-1).
// - lower_state  [i/o] : State of the lower filter, given in Q(-1).
// - hp_data_out  [o]   : Output audio data of the upper half of the spectrum.
//                        The length is |data_length| / 2.
// - lp_data_out  [o]   : Output audio data of the lower half of the spectrum.
//                        The length is |data_length| / 2.
function splitFilter(data_in, data_length, self, frequency_band, hp_data_out, lp_data_out) {
    var i;
    var half_length = data_length >> 1;  // Downsampling by 2.
    var tmp_out = new Int16Array(new ArrayBuffer(2));

    // All-pass filtering upper branch.
    allPassFilter(data_in, 0, half_length, kAllPassCoefsQ15[0], self.upper_state, frequency_band,
                  hp_data_out);

    // All-pass filtering lower branch.
    allPassFilter(data_in, 1, half_length, kAllPassCoefsQ15[1], self.lower_state, frequency_band,
                  lp_data_out);

    // Make LP and HP signals.
    for (i = 0; i < half_length; i++) {
        tmp_out[0] = hp_data_out[i];
        hp_data_out[i] -= lp_data_out[i];
        lp_data_out[i] += tmp_out[0];
    }
}

// Calculates the energy of |data_in| in dB, and also updates an overall
// |total_energy| if necessary.
//
// - data_in      [i]   : Input audio data for energy calculation.
// - data_length  [i]   : Length of input data.
// - offset       [i]   : Offset value added to |log_energy|.
// - total_energy [i/o] : An external energy updated with the energy of
//                        |data_in|.
//                        NOTE: |total_energy| is only updated if
//                        |total_energy| <= |kMinEnergy|.
// - log_energy   [o]   : 10 * log10("energy of |data_in|") given in Q4.
function logOfEnergy(data_in, data_length, offset, total_energy, log_energy) {
    // |tot_rshifts| accumulates the number of right shifts performed on |energy|.
    var tot_rshifts = 0;
    // The |energy| will be normalized to 15 bits. We use unsigned integer because
    // we eventually will mask out the fractional part.
    var energy = 0;
    var en = signalProcess.WebRtcSpl_Energy(data_in, data_length, tot_rshifts);
    energy = en.energy;
    tot_rshifts = en.scale_factor;

    if (energy != 0) {
        // By construction, normalizing to 15 bits is equivalent with 17 leading
        // zeros of an unsigned 32 bit value.
        var normalizing_rshifts = 17 - signalProcess.WebRtcSpl_NormU32(energy);
        // In a 15 bit representation the leading bit is 2^14. log2(2^14) in Q10 is
        // (14 << 10), which is what we initialize |log2_energy| with. For a more
        // detailed derivations, see below.
        var log2_energy = kLogEnergyIntPart;

        tot_rshifts += normalizing_rshifts;
        // Normalize |energy| to 15 bits.
        // |tot_rshifts| is now the total number of right shifts performed on
        // |energy| after normalization. This means that |energy| is in
        // Q(-tot_rshifts).
        if (normalizing_rshifts < 0) {
            energy <<= -normalizing_rshifts;
        } else {
            energy >>= normalizing_rshifts;
        }

        // Calculate the energy of |data_in| in dB, in Q4.
        //
        // 10 * log10("true energy") in Q4 = 2^4 * 10 * log10("true energy") =
        // 160 * log10(|energy| * 2^|tot_rshifts|) =
        // 160 * log10(2) * log2(|energy| * 2^|tot_rshifts|) =
        // 160 * log10(2) * (log2(|energy|) + log2(2^|tot_rshifts|)) =
        // (160 * log10(2)) * (log2(|energy|) + |tot_rshifts|) =
        // |kLogConst| * (|log2_energy| + |tot_rshifts|)
        //
        // We know by construction that |energy| is normalized to 15 bits. Hence,
        // |energy| = 2^14 + frac_Q15, where frac_Q15 is a fractional part in Q15.
        // Further, we'd like |log2_energy| in Q10
        // log2(|energy|) in Q10 = 2^10 * log2(2^14 + frac_Q15) =
        // 2^10 * log2(2^14 * (1 + frac_Q15 * 2^-14)) =
        // 2^10 * (14 + log2(1 + frac_Q15 * 2^-14)) ~=
        // (14 << 10) + 2^10 * (frac_Q15 * 2^-14) =
        // (14 << 10) + (frac_Q15 * 2^-4) = (14 << 10) + (frac_Q15 >> 4)
        //
        // Note that frac_Q15 = (|energy| & 0x00003FFF)

        // Calculate and add the fractional part to |log2_energy|.
        log2_energy += ((energy & 0x00003FFF) >> 4);

        // |kLogConst| is in Q9, |log2_energy| in Q10 and |tot_rshifts| in Q0.
        // Note that we in our derivation above have accounted for an output in Q4.
        log_energy = (((kLogConst * log2_energy) >> 19) + ((tot_rshifts * kLogConst) >> 9));

        if (log_energy < 0) {
            log_energy = 0;
        }
    } else {
        log_energy = offset;
        return {log_energy, total_energy};
    }

    log_energy += offset;

    // Update the approximate |total_energy| with the energy of |data_in|, if
    // |total_energy| has not exceeded |kMinEnergy|. |total_energy| is used as an
    // energy indicator in WebRtcVad_GmmProbability() in vad_core.c.
    if (total_energy <= kMinEnergy) {
        if (tot_rshifts >= 0) {
            // We know by construction that the |energy| > |kMinEnergy| in Q0, so add
            // an arbitrary value such that |total_energy| exceeds |kMinEnergy|.
            total_energy += kMinEnergy + 1;
        } else {
            // By construction |energy| is represented by 15 bits, hence any number of
            // right shifted |energy| will fit in an int16_t. In addition, adding the
            // value to |total_energy| is wrap around safe as long as
            // |kMinEnergy| < 8192.
            total_energy += (energy >> -tot_rshifts);  // Q0.
        }
    }
    return {log_energy, total_energy};
}

function WebRtcVad_CalculateFeatures(self, data_in, data_length, features) {
    var total_energy = 0;
    // We expect |data_length| to be 80, 160 or 240 samples, which corresponds to
    // 10, 20 or 30 ms in 8 kHz. Therefore, the intermediate downsampled data will
    // have at most 120 samples after the first split and at most 60 samples after
    // the second split.
    var hp_120 = new Int16Array(new ArrayBuffer(2*120)), lp_120  = new Int16Array(new ArrayBuffer(2*120));
    var hp_60 = new Int16Array(new ArrayBuffer(2*60)), lp_60 = new Int16Array(new ArrayBuffer(2*60));
    var half_data_length = data_length >> 1;
    var length = half_data_length;  // |data_length| / 2, corresponds to
    // bandwidth = 2000 Hz after downsampling.
    hp_60.fill(0, 0);
    lp_60.fill(0, 0);
    hp_120.fill(0, 0);
    lp_120.fill(0, 0);
    // Initialize variables for the first SplitFilter().
    var frequency_band = 0;
    var in_ptr = data_in;  // [0 - 4000] Hz.
    var hp_out_ptr = hp_120;  // [2000 - 4000] Hz.
    var lp_out_ptr = lp_120;  // [0 - 2000] Hz.

    if(data_length > 240) return;
    //if(4 < 6 - 1)  // Checking maximum |frequency_band|.

    // Split at 2000 Hz and downsample.
    splitFilter(in_ptr, data_length, self, frequency_band, hp_out_ptr, lp_out_ptr);

    // For the upper band (2000 Hz - 4000 Hz) split at 3000 Hz and downsample.
    frequency_band = 1;
    in_ptr = hp_120;  // [2000 - 4000] Hz.
    hp_out_ptr = hp_60;  // [3000 - 4000] Hz.
    lp_out_ptr = lp_60;  // [2000 - 3000] Hz.
    splitFilter(in_ptr, length, self, frequency_band, hp_out_ptr, lp_out_ptr);

    // Energy in 3000 Hz - 4000 Hz.
    length >>= 1;  // |data_length| / 4 <=> bandwidth = 1000 Hz.

    var logEnergy = logOfEnergy(hp_60, length, kOffsetVector[5], total_energy, features[5]);
    total_energy = logEnergy.total_energy;
    features[5] = logEnergy.log_energy;

    // Energy in 2000 Hz - 3000 Hz.
    logEnergy = logOfEnergy(lp_60, length, kOffsetVector[4], total_energy, features[4]);
    total_energy = logEnergy.total_energy;
    features[4] = logEnergy.log_energy;

    // For the lower band (0 Hz - 2000 Hz) split at 1000 Hz and downsample.
    frequency_band = 2;
    in_ptr = lp_120;  // [0 - 2000] Hz.
    hp_out_ptr = hp_60;  // [1000 - 2000] Hz.
    lp_out_ptr = lp_60;  // [0 - 1000] Hz.
    length = half_data_length;  // |data_length| / 2 <=> bandwidth = 2000 Hz.
    splitFilter(in_ptr, length, self, frequency_band, hp_out_ptr, lp_out_ptr);

    // Energy in 1000 Hz - 2000 Hz.
    length >>= 1;  // |data_length| / 4 <=> bandwidth = 1000 Hz.
    logEnergy = logOfEnergy(hp_60, length, kOffsetVector[3], total_energy, features[3]);
    total_energy = logEnergy.total_energy;
    features[3] = logEnergy.log_energy;

    // For the lower band (0 Hz - 1000 Hz) split at 500 Hz and downsample.
    frequency_band = 3;
    in_ptr = lp_60;  // [0 - 1000] Hz.
    hp_out_ptr = hp_120;  // [500 - 1000] Hz.
    lp_out_ptr = lp_120;  // [0 - 500] Hz.
    splitFilter(in_ptr, length, self, frequency_band, hp_out_ptr, lp_out_ptr);

    // Energy in 500 Hz - 1000 Hz.
    length >>= 1;  // |data_length| / 8 <=> bandwidth = 500 Hz.
    logEnergy = logOfEnergy(hp_120, length, kOffsetVector[2], total_energy, features[2]);
    total_energy = logEnergy.total_energy;
    features[2] = logEnergy.log_energy;

    // For the lower band (0 Hz - 500 Hz) split at 250 Hz and downsample.
    frequency_band = 4;
    in_ptr = lp_120;  // [0 - 500] Hz.
    hp_out_ptr = hp_60;  // [250 - 500] Hz.
    lp_out_ptr = lp_60;  // [0 - 250] Hz.
    splitFilter(in_ptr, length, self, frequency_band, hp_out_ptr, lp_out_ptr);

    // Energy in 250 Hz - 500 Hz.
    length >>= 1;  // |data_length| / 16 <=> bandwidth = 250 Hz.
    logEnergy = logOfEnergy(hp_60, length, kOffsetVector[1], total_energy, features[1]);
    total_energy = logEnergy.total_energy;
    features[1] = logEnergy.log_energy;

    // Remove 0 Hz - 80 Hz, by high pass filtering the lower band.
    highPassFilter(lp_60, length, self.hp_filter_state, hp_120);

    // Energy in 80 Hz - 250 Hz.
    logEnergy = logOfEnergy(hp_120, length, kOffsetVector[0], total_energy, features[0]);
    total_energy = logEnergy.total_energy;
    features[0] = logEnergy.log_energy;

    return total_energy;
}



module.exports = {
    WebRtcVad_CalculateFeatures
}

