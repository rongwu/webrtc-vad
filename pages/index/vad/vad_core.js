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
    WebRtcVad_Downsampling,
    WebRtcVad_FindMinimum
 } from "./vad_sp.js";
import { WebRtcVad_GaussianProbability } from "./vad_gmm.js";
import { WebRtcVad_CalculateFeatures } from "./vad_filterbank.js";
var signalProcess = require("./signal_processing.js");

const kAllPassCoefsQ13 = [5243, 1392];  // Q13.

// Thresholds for different frame lengths (10 ms, 20 ms and 30 ms).
// Mode 0, Quality.
const kOverHangMax1Q = [8, 4, 3];
const kOverHangMax2Q = [14, 7, 5];
const kLocalThresholdQ = [24, 21, 24];
const kGlobalThresholdQ = [57, 48, 57];
// Mode 1, Low bitrate.
const kOverHangMax1LBR = [8, 4, 3];
const kOverHangMax2LBR = [14, 7, 5];
const kLocalThresholdLBR = [37, 32, 37];
const kGlobalThresholdLBR = [100, 80, 100];
// Mode 2, Aggressive.
const kOverHangMax1AGG = [6, 3, 2];
const kOverHangMax2AGG = [9, 5, 3];
const kLocalThresholdAGG = [82, 78, 82];
const kGlobalThresholdAGG = [285, 260, 285];
// Mode 3, Very aggressive.
const kOverHangMax1VAG = [6, 3, 2];
const kOverHangMax2VAG = [9, 5, 3];
const kLocalThresholdVAG = [94, 94, 94];
const kGlobalThresholdVAG = [1100, 1050, 1100];

// Spectrum Weighting
const kSpectrumWeight = [6, 8, 10, 12, 14, 16];
const kNoiseUpdateConst = 655; // Q15
const kSpeechUpdateConst = 6554; // Q15
const kBackEta = 154; // Q8

const kNumChannels = 6; // Number of frequency bands (named channels).
const kNumGaussians = 2; // Number of Gaussians per channel in the GMM.
const kTableSize =  kNumChannels * kNumGaussians;
const kMinEnergy = 10;
// Minimum difference between the two models, Q5
const kMinimumDifference = [544, 544, 576, 576, 576, 576];
// Upper limit of mean value for speech model, Q7
const kMaximumSpeech = [11392, 11392, 11520, 11520, 11520, 11520];
// Minimum value for mean value
const kMinimumMean = [640, 768];
// Upper limit of mean value for noise model, Q7
const kMaximumNoise = [9216, 9088, 8960, 8832, 8704, 8576];
// Start values for the Gaussian models, Q7
// Weights for the two Gaussians for the six channels (noise)
const kNoiseDataWeights = [34, 62, 72, 66, 53, 25, 94, 66, 56, 62, 75, 103];
// Weights for the two Gaussians for the six channels (speech)
const kSpeechDataWeights = [48, 82, 45, 87, 50, 47, 80, 46, 83, 41, 78, 81];
// Means for the two Gaussians for the six channels (noise)
const kNoiseDataMeans = [6738, 4892, 7065, 6715, 6771, 3369, 7646, 3863, 7820, 7266, 5020, 4362];
// Means for the two Gaussians for the six channels (speech)
const kSpeechDataMeans = [8306, 10085, 10078, 11823, 11843, 6309, 9473, 9571, 10879, 7581, 8180, 7483];
// Stds for the two Gaussians for the six channels (noise)
const kNoiseDataStds = [378, 1064, 493, 582, 688, 593, 474, 697, 475, 688, 421, 455];
// Stds for the two Gaussians for the six channels (speech)
const kSpeechDataStds = [555, 505, 567, 524, 585, 1231, 509, 828, 492, 1540, 1079, 850];

// Constants used in GmmProbability().
//
// Maximum number of counted speech (VAD = 1) frames in a row.
const kMaxSpeechFrames = 6;
// Minimum standard deviation for both speech and noise.
const kMinStd = 384;


// Constants in WebRtcVad_InitCore().
// Default aggressiveness mode.
const kDefaultMode = 0;
const kInitCheck = 42;


// Calculates the weighted average w.r.t. number of Gaussians. The |data| are
// updated with an |offset| before averaging.
//
// - data     [i/o] : Data to average.
// - offset   [i]   : An offset added to |data|.
// - weights  [i]   : Weights used for averaging.
//
// returns          : The weighted average.
function WeightedAverage(data, offset, weights, channel) {
    var k;
    var weighted_average = 0;

    for (k = 0; k < kNumChannels; k++) {
        var i = k * kNumChannels + channel; 
        data[i] += offset;
        weighted_average += data[i] * weights[i];
    }
    return weighted_average;
}

// An s16 x s32 -> s32 multiplication that's allowed to overflow. (It's still
// undefined behavior, so not a good idea; this just makes UBSan ignore the
// violation, so that our old code can continue to do what it's always been
// doing.)
function OverflowingMulS16ByS32ToS32(a, b) {
    return a * b;
}

// Calculates the probabilities for both speech and background noise using
// Gaussian Mixture Models (GMM). A hypothesis-test is performed to decide which
// type of signal is most probable.
//
// - self           [i/o] : Pointer to VAD instance
// - features       [i]   : Feature vector of length |kNumChannels|
//                          = log10(energy in frequency band)
// - total_power    [i]   : Total power in audio frame.
// - frame_length   [i]   : Number of input samples
//
// - returns              : the VAD decision (0 - noise, 1 - speech).
function GmmProbability(self, features, total_power, frame_length) {
   var channel, k;
    var feature_minimum = new Int16Array(1);
    var h0 = new Int16Array(1), h1 = new Int16Array(1);
    var log_likelihood_ratio = new Int16Array(1);
    var vadflag = new Int16Array(1);vadflag[0] = 0;
    var shifts_h0 = new Int16Array(1), shifts_h1 = new Int16Array(1);
    var tmp_s16 = new Int16Array(1), 
    tmp1_s16 = new Int16Array(1), tmp2_s16 = new Int16Array(1);
    var diff = new Int16Array(1);
    var gaussian;
    var nmk = new Int16Array(1), nmk2 = new Int16Array(1), 
    nmk3 = new Int16Array(1), smk = new Int16Array(1), 
    smk2 = new Int16Array(1), nsk = new Int16Array(1), 
    ssk = new Int16Array(1);
    var delt = new Int16Array(1), ndelt = new Int16Array(1);
    var maxspe = new Int16Array(1), maxmu = new Int16Array(1);
    var deltaN = new Int16Array(kTableSize), 
    deltaS = new Int16Array(kTableSize);
    var ngprvec = new Int16Array(kTableSize);  // Conditional probability = 0.
    var sgprvec = new Int16Array(kTableSize);  // Conditional probability = 0.
    var h0_test = new Int32Array(1), h1_test = new Int32Array(1);
    var tmp1_s32 = new Int32Array(1), tmp2_s32 = new Int32Array(1);
    var sum_log_likelihood_ratios = new Int32Array(1);
    var noise_global_mean = new Int32Array(1), 
    speech_global_mean = new Int32Array(1);
    var noise_probability = new Int32Array(2), 
    speech_probability  = new Int32Array(2);
    var overhead1 = new Int16Array(1), 
    overhead2 = new Int16Array(1), 
    individualTest = new Int16Array(1), 
    totalTest = new Int16Array(1);

    sum_log_likelihood_ratios[0] = 0;
    noise_probability.fill(0,0);
    speech_probability.fill(0,0);

    deltaN.fill(0,0);
    deltaS.fill(0,0);
    ngprvec.fill(0,0);
    sgprvec.fill(0,0);

    // Set various thresholds based on frame lengths (80, 160 or 240 samples).
    if (frame_length == 80) {
        overhead1[0] = self.over_hang_max_1[0];
        overhead2[0] = self.over_hang_max_2[0];
        individualTest[0] = self.individual[0];
        totalTest[0] = self.total[0];
    } else if (frame_length == 160) {
        overhead1[0] = self.over_hang_max_1[1];
        overhead2[0] = self.over_hang_max_2[1];
        individualTest[0] = self.individual[1];
        totalTest[0] = self.total[1];
    } else {
        overhead1[0] = self.over_hang_max_1[2];
        overhead2[0] = self.over_hang_max_2[2];
        individualTest[0] = self.individual[2];
        totalTest[0] = self.total[2];
    }

    if (total_power > kMinEnergy) {
        // The signal power of current frame is large enough for processing. The
        // processing consists of two parts:
        // 1) Calculating the likelihood of speech and thereby a VAD decision.
        // 2) Updating the underlying model, w.r.t., the decision made.

        // The detection scheme is an LRT with hypothesis
        // H0: Noise
        // H1: Speech
        //
        // We combine a global LRT with local tests, for each frequency sub-band,
        // here defined as |channel|.
        for (channel = 0; channel < kNumChannels; channel++) {
            // For each channel we model the probability with a GMM consisting of
            // |kNumGaussians|, with different means and standard deviations depending
            // on H0 or H1.
            h0_test[0] = 0;
            h1_test[0] = 0;
            for (k = 0; k < kNumGaussians; k++) {
                gaussian = channel + k * kNumChannels;
                // Probability under H0, that is, probability of frame being noise.
                // Value given in Q27 = Q7 * Q20.
                tmp1_s32[0] = WebRtcVad_GaussianProbability(features[channel],
                                                         self.noise_means[gaussian],
                                                         self.noise_stds[gaussian],
                                                         deltaN, gaussian);
                
                noise_probability[k] = kNoiseDataWeights[gaussian] * tmp1_s32[0];
                h0_test[0] += noise_probability[k];  // Q27

                // Probability under H1, that is, probability of frame being speech.
                // Value given in Q27 = Q7 * Q20.
                tmp1_s32[0] = WebRtcVad_GaussianProbability(features[channel],
                                                         self.speech_means[gaussian],
                                                         self.speech_stds[gaussian],
                                                         deltaS, gaussian);
                speech_probability[k] = kSpeechDataWeights[gaussian] * tmp1_s32[0];
                h1_test[0] += speech_probability[k];  // Q27
            }

            // Calculate the log likelihood ratio: log2(Pr{X|H1} / Pr{X|H1}).
            // Approximation:
            // log2(Pr{X|H1} / Pr{X|H1}) = log2(Pr{X|H1}*2^Q) - log2(Pr{X|H1}*2^Q)
            //                           = log2(h1_test) - log2(h0_test[0])
            //                           = log2(2^(31-shifts_h1)*(1+b1))
            //                             - log2(2^(31-shifts_h0)*(1+b0))
            //                           = shifts_h0 - shifts_h1
            //                             + log2(1+b1) - log2(1+b0)
            //                          ~= shifts_h0 - shifts_h1
            //
            // Note that b0 and b1 are values less than 1, hence, 0 <= log2(1+b0) < 1.
            // Further, b0 and b1 are independent and on the average the two terms
            // cancel.
            shifts_h0[0] = signalProcess.WebRtcSpl_NormW32(h0_test[0]);
            shifts_h1[0] = signalProcess.WebRtcSpl_NormW32(h1_test[0]);
            if (h0_test[0] == 0) {
                shifts_h0[0] = 31;
            }
            if (h1_test[0] == 0) {
                shifts_h1[0] = 31;
            }
            log_likelihood_ratio[0] = shifts_h0[0] - shifts_h1[0];

            // Update |sum_log_likelihood_ratios| with spectrum weighting. This is
            // used for the global VAD decision.
            sum_log_likelihood_ratios[0] += (log_likelihood_ratio[0] * kSpectrumWeight[channel]);

            // Local VAD decision.
            if ((log_likelihood_ratio[0] * 4) > individualTest[0]) {
                vadflag[0] = 1;
            }

            // TODO(bjornv): The conditional probabilities below are applied on the
            // hard coded number of Gaussians set to two. Find a way to generalize.
            // Calculate local noise probabilities used later when updating the GMM.
            h0[0] = (h0_test[0] >> 12);  // Q15
            if (h0[0] > 0) {
                // High probability of noise. Assign conditional probabilities for each
                // Gaussian in the GMM.
                tmp1_s32[0] = (noise_probability[0] & 0xFFFFF000) << 2;  // Q29
                ngprvec[channel] = signalProcess.WebRtcSpl_DivW32W16(tmp1_s32[0], h0[0]);  // Q14
                ngprvec[channel + kNumChannels] = 16384 - ngprvec[channel];
            } else {
                // Low noise probability. Assign conditional probability 1 to the first
                // Gaussian and 0 to the rest (which is already set at initialization).
                ngprvec[channel] = 16384;
            }

            // Calculate local speech probabilities used later when updating the GMM.
            h1[0] = (h1_test[0] >> 12);  // Q15
            if (h1[0] > 0) {
                // High probability of speech. Assign conditional probabilities for each
                // Gaussian in the GMM. Otherwise use the initialized values, i.e., 0.
                tmp1_s32[0] = (speech_probability[0] & 0xFFFFF000) << 2;  // Q29
                sgprvec[channel] = signalProcess.WebRtcSpl_DivW32W16(tmp1_s32[0], h1[0]);  // Q14
                sgprvec[channel + kNumChannels] = 16384 - sgprvec[channel];
            }
        }

        // Make a global VAD decision.
        vadflag[0] |= (sum_log_likelihood_ratios[0] >= totalTest[0]);

        // Update the model parameters.
        maxspe[0] = 12800;
        for (channel = 0; channel < kNumChannels; channel++) {

            // Get minimum value in past which is used for long term correction in Q4.
            feature_minimum[0] = WebRtcVad_FindMinimum(self, features[channel], channel);

            // Compute the "global" mean, that is the sum of the two means weighted.
            noise_global_mean[0] = WeightedAverage(self.noise_means, 0,
                                                kNoiseDataWeights, channel);
            tmp1_s16[0] = (noise_global_mean[0] >> 6);  // Q8

            for (k = 0; k < kNumGaussians; k++) {
                gaussian = channel + k * kNumChannels;

                nmk[0] = self.noise_means[gaussian];
                smk[0] = self.speech_means[gaussian];
                nsk[0] = self.noise_stds[gaussian];
                ssk[0] = self.speech_stds[gaussian];

                // Update noise mean vector if the frame consists of noise only.
                nmk2[0] = nmk[0];
                if (!vadflag[0]) {
                    // deltaN = (x-mu)/sigma^2
                    // ngprvec[k] = |noise_probability[k]| /
                    //   (|noise_probability[0]| + |noise_probability[1]|)

                    // (Q14 * Q11 >> 11) = Q14.
                    delt[0] = ((ngprvec[gaussian] * deltaN[gaussian]) >> 11);
                    // Q7 + (Q14 * Q15 >> 22) = Q7.
                    nmk2[0] = nmk[0] + ((delt[0] * kNoiseUpdateConst) >> 22);
                }

                // Long term correction of the noise mean.
                // Q8 - Q8 = Q8.
                ndelt[0] = (feature_minimum[0] << 4) - tmp1_s16[0];
                // Q7 + (Q8 * Q8) >> 9 = Q7.
                nmk3[0] = nmk2[0] + ((ndelt[0] * kBackEta) >> 9);

                // Control that the noise mean does not drift to much.
                tmp_s16[0] = ((k + 5) << 7);
                if (nmk3[0] < tmp_s16[0]) {
                    nmk3[0] = tmp_s16[0];
                }
                tmp_s16[0] = ((72 + k - channel) << 7);
                if (nmk3[0] > tmp_s16[0]) {
                    nmk3[0] = tmp_s16[0];
                }
                self.noise_means[gaussian] = nmk3[0];

                if (vadflag[0]) {
                    // Update speech mean vector:
                    // |deltaS| = (x-mu)/sigma^2
                    // sgprvec[k] = |speech_probability[k]| /
                    //   (|speech_probability[0]| + |speech_probability[1]|)

                    // (Q14 * Q11) >> 11 = Q14.
                    delt[0] = ((sgprvec[gaussian] * deltaS[gaussian]) >> 11);
                    // Q14 * Q15 >> 21 = Q8.
                    tmp_s16[0] = ((delt[0] * kSpeechUpdateConst) >> 21);
                    // Q7 + (Q8 >> 1) = Q7. With rounding.
                    smk2[0] = smk[0] + ((tmp_s16[0] + 1) >> 1);

                    // Control that the speech mean does not drift to much.
                    maxmu[0] = maxspe[0] + 640;
                    if (smk2[0] < kMinimumMean[k]) {
                        smk2[0] = kMinimumMean[k];
                    }
                    if (smk2[0] > maxmu[0]) {
                        smk2[0] = maxmu[0];
                    }
                    self.speech_means[gaussian] = smk2[0];  // Q7.

                    // (Q7 >> 3) = Q4. With rounding.
                    tmp_s16[0] = ((smk[0] + 4) >> 3);

                    tmp_s16[0] = features[channel] - tmp_s16[0];  // Q4
                    // (Q11 * Q4 >> 3) = Q12.
                    tmp1_s32[0] = (deltaS[gaussian] * tmp_s16[0]) >> 3;
                    tmp2_s32[0] = tmp1_s32[0] - 4096;
                    tmp_s16[0] = sgprvec[gaussian] >> 2;
                    // (Q14 >> 2) * Q12 = Q24.
                    tmp1_s32[0] = tmp_s16[0] * tmp2_s32[0];

                    tmp2_s32[0] = tmp1_s32[0] >> 4;  // Q20

                    // 0.1 * Q20 / Q7 = Q13.
                    if (tmp2_s32[0] > 0) {
                        tmp_s16[0] = signalProcess.WebRtcSpl_DivW32W16(tmp2_s32[0], ssk[0] * 10);
                    } else {
                        tmp_s16[0] = signalProcess.WebRtcSpl_DivW32W16(-tmp2_s32[0], ssk[0] * 10);
                        tmp_s16[0] = -tmp_s16[0];
                    }
                    // Divide by 4 giving an update factor of 0.025 (= 0.1 / 4).
                    // Note that division by 4 equals shift by 2, hence,
                    // (Q13 >> 8) = (Q13 >> 6) / 4 = Q7.
                    tmp_s16[0] += 128;  // Rounding.
                    ssk[0] += (tmp_s16[0] >> 8);
                    if (ssk[0] < kMinStd) {
                        ssk[0] = kMinStd;
                    }
                    self.speech_stds[gaussian] = ssk[0];
                } else {
                    // Update GMM variance vectors.
                    // deltaN * (features[channel] - nmk) - 1
                    // Q4 - (Q7 >> 3) = Q4.
                    tmp_s16[0] = features[channel] - (nmk[0] >> 3);
                    // (Q11 * Q4 >> 3) = Q12.
                    tmp1_s32[0] = (deltaN[gaussian] * tmp_s16[0]) >> 3;
                    tmp1_s32[0] -= 4096;

                    // (Q14 >> 2) * Q12 = Q24.
                    tmp_s16[0] = (ngprvec[gaussian] + 2) >> 2;
                    tmp2_s32[0] = OverflowingMulS16ByS32ToS32(tmp_s16[0], tmp1_s32[0]);
                    // Q20  * approx 0.001 (2^-10=0.0009766), hence,
                    // (Q24 >> 14) = (Q24 >> 4) / 2^10 = Q20.
                    tmp1_s32[0] = tmp2_s32[0] >> 14;

                    // Q20 / Q7 = Q13.
                    if (tmp1_s32[0] > 0) {
                        tmp_s16[0] = signalProcess.WebRtcSpl_DivW32W16(tmp1_s32[0], nsk[0]);
                    } else {
                        tmp_s16[0] = signalProcess.WebRtcSpl_DivW32W16(-tmp1_s32[0], nsk[0]);
                        tmp_s16[0] = -tmp_s16[0];
                    }
                    tmp_s16[0] += 32;  // Rounding
                    nsk[0] += tmp_s16[0] >> 6;  // Q13 >> 6 = Q7.
                    if (nsk[0] < kMinStd) {
                        nsk[0] = kMinStd;
                    }
                    self.noise_stds[gaussian] = nsk[0];
                }
            }

            // Separate models if they are too close.
            // |noise_global_mean| in Q14 (= Q7 * Q7).
            noise_global_mean[0] = WeightedAverage(self.noise_means, 0,
                kNoiseDataWeights, channel);

            // |speech_global_mean| in Q14 (= Q7 * Q7).
            speech_global_mean[0] = WeightedAverage(self.speech_means, 0,
                                                 kSpeechDataWeights, channel);

            // |diff| = "global" speech mean - "global" noise mean.
            // (Q14 >> 9) - (Q14 >> 9) = Q5.
            diff[0] = (speech_global_mean[0] >> 9) -
                   (noise_global_mean[0] >> 9);
            if (diff[0] < kMinimumDifference[channel]) {
                tmp_s16[0] = kMinimumDifference[channel] - diff[0];

                // |tmp1_s16| = ~0.8 * (kMinimumDifference - diff) in Q7.
                // |tmp2_s16| = ~0.2 * (kMinimumDifference - diff) in Q7.
                tmp1_s16[0] = ((13 * tmp_s16[0]) >> 2);
                tmp2_s16[0] = ((3 * tmp_s16[0]) >> 2);

                // Move Gaussian means for speech model by |tmp1_s16| and update
                // |speech_global_mean|. Note that |self.speech_means[channel]| is
                // changed after the call.
                speech_global_mean[0] = WeightedAverage(self.speech_means,
                                                     tmp1_s16[0],
                                                     kSpeechDataWeights, channel);

                // Move Gaussian means for noise model by -|tmp2_s16| and update
                // |noise_global_mean|. Note that |self.noise_means[channel]| is
                // changed after the call.
                noise_global_mean[0] = WeightedAverage(self.noise_means,
                                                    -tmp2_s16[0],
                                                    kNoiseDataWeights, channel);
            }

            // Control that the speech & noise means do not drift to much.
            maxspe[0] = kMaximumSpeech[channel];
            tmp2_s16[0] = (speech_global_mean[0] >> 7);
            if (tmp2_s16[0] > maxspe[0]) {
                // Upper limit of speech model.
                tmp2_s16[0] -= maxspe[0];

                for (k = 0; k < kNumGaussians; k++) {
                    self.speech_means[channel + k * kNumChannels] -= tmp2_s16[0];
                }
            }

            tmp2_s16[0] = (noise_global_mean[0] >> 7);
            if (tmp2_s16[0] > kMaximumNoise[channel]) {
                tmp2_s16[0] -= kMaximumNoise[channel];

                for (k = 0; k < kNumGaussians; k++) {
                    self.noise_means[channel + k * kNumChannels] -= tmp2_s16[0];
                }
            }
        }
        self.frame_counter++;
    }

    // Smooth with respect to transition hysteresis.
    if (!vadflag[0]) {
        if (self.over_hang > 0) {
            vadflag[0] = 2 + self.over_hang;
            self.over_hang--;
        }
        self.num_of_speech = 0;
    } else {
        self.num_of_speech++;
        if (self.num_of_speech > kMaxSpeechFrames) {
            self.num_of_speech = kMaxSpeechFrames;
            self.over_hang = overhead2[0];
        } else {
            self.over_hang = overhead1[0];
        }
    }
    return vadflag[0];
}

// Initialize the VAD. Set aggressiveness mode to default value.
function WebRtcVad_InitCore(self, mode) {
    // Read initial PDF parameters.
    for(var i = 0; i < kTableSize; i++){
        self.noise_means[i] = kNoiseDataMeans[i];
        self.speech_means[i] = kSpeechDataMeans[i];
        self.noise_stds[i] = kNoiseDataStds[i];
        self.speech_stds[i] = kSpeechDataStds[i];
    }
      // Initialize Index and Minimum value vectors.
      
    self.index_vector.fill(0,0);
    self.low_value_vector.fill(10000,0);
  
      // Initialize mean value memory, for WebRtcVad_FindMinimum().
    for (i = 0; i < kNumChannels; i++) {
        self.mean_value[i] = 1600;
    }
    WebRtcVad_set_mode_core(self, mode);
    self.init_flag = 42;

    return 0;
}

// Set aggressiveness mode
function WebRtcVad_set_mode_core(self, mode) {
    switch (mode) {
        case 0:
            // Quality mode.
            self.over_hang_max_1 = kOverHangMax1Q;
            self.over_hang_max_2 = kOverHangMax2Q;
            self.individual = kLocalThresholdQ;
            self.total = kGlobalThresholdQ;
            break;
        case 1:
            // Low bitrate mode.
            self.over_hang_max_1 = kOverHangMax1LBR;
            self.over_hang_max_2 = kOverHangMax2LBR;
            self.individual = kLocalThresholdLBR;
            self.total = kGlobalThresholdLBR;
            break;
        case 2:
            // Aggressive mode.
            self.over_hang_max_1 = kOverHangMax1AGG;
            self.over_hang_max_2 = kOverHangMax2AGG;
            self.individual = kLocalThresholdAGG;
            self.total = kGlobalThresholdAGG;
            break;
        case 3:
            // Very aggressive mode.
            self.over_hang_max_1 = kOverHangMax1VAG;
            self.over_hang_max_2 = kOverHangMax2VAG;
            self.individual = kLocalThresholdVAG;
            self.total = kGlobalThresholdVAG;
            break;    
    }
}

// Calculate VAD decision by first extracting feature values and then calculate
// probability for both speech and background noise.



function WebRtcVad_CalcVad32khz(inst, speech_frame) {
    
    var speechWB = new Int16Array(480); // Downsampled speech frame: 960 samples (30ms in SWB)
    var speechNB = new Int16Array(240); // Downsampled speech frame: 480 samples (30ms in WB)
    var frame_length = speech_frame.length/2;
    var dfs = inst.downsampling_filter_states.subarray(0);
    // Downsample signal 32->16->8 before doing VAD
    WebRtcVad_Downsampling(speech_frame, speechWB, dfs, frame_length);
    inst.downsampling_filter_states.set(2, dfs);

    var len = frame_length / 2;
    WebRtcVad_Downsampling(speechWB, speechNB, inst.downsampling_filter_states, len);
    //len /= 2;

    // Do VAD on an 8 kHz signal
    vad = WebRtcVad_CalcVad8khz(inst, speechNB, len);

    return vad;
}

function WebRtcVad_CalcVad16khz(inst, speech_frame) {
    
    var half_length = speech_frame.length/2;
    var speechNB = new Int16Array(240); // Downsampled speech frame: 480 samples (30ms in WB)
    // Wideband: Downsample signal before doing VAD
    var filter_state = inst.downsampling_filter_states;
    WebRtcVad_Downsampling(speech_frame, speechNB, filter_state, half_length);

    var vad = WebRtcVad_CalcVad8khz(inst, speechNB, half_length);
    //console.log(vad); 
    return vad;
}

function WebRtcVad_CalcVad8khz(inst, speech_frame, frame_length) {
    var feature_vector = [0,0,0,0,0,0], total_power;

    // Get power in the bands
    total_power = WebRtcVad_CalculateFeatures(inst, speech_frame, frame_length, feature_vector);
    // Make a VAD
    inst.vad = GmmProbability(inst, feature_vector, total_power, frame_length);
    
    return inst.vad;
}

module.exports = {
    kTableSize,
    kNumChannels,
    WebRtcVad_CalcVad32khz,
    WebRtcVad_CalcVad16khz,
    WebRtcVad_CalcVad8khz,
    WebRtcVad_InitCore
}

