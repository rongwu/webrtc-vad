/*
 *  Copyright (c) 2011 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

var signalProcess = require("./signal_processing.js");

var kCompVar = 22005;
var kLog2Exp = 5909;  // log2(exp(1)) in Q12.

// For a normal distribution, the probability of |input| is calculated and
// returned (in Q20). The formula for normal distributed probability is
//
// 高斯正太分布的概率计算公式为
// 1 / s * exp(-(x - m)^2 / (2 * s^2))  
//
// where the parameters are given in the following Q domains:
// m = |mean| (Q7)
// s = |std| (Q7)
// x = |input| (Q4)
// in addition to the probability we output |delta| (in Q11) used when updating
// the noise/speech model.
function WebRtcVad_GaussianProbability(input, mean, std, delta, index) {
    var tmp16 = new Int16Array(new ArrayBuffer(2)), 
    inv_std = new Int16Array(new ArrayBuffer(2)), 
    inv_std2 = new Int16Array(new ArrayBuffer(2)), 
    exp_value = new Int16Array(new ArrayBuffer(2));
    exp_value[0] = 0;
    var tmp32;

    // Calculate |inv_std| = 1 / s, in Q10.
    // 131072 = 1 in Q17, and (|std| >> 1) is for rounding instead of truncation.
    // Q-domain: Q17 / Q7 = Q10.
    //求方差倒数 1/s，其做法是对小数进行四舍五入（rounding）而不是直接截断（truncation）, 所以加了(std >> 1), 即在Q7的std的一半，也就是0.5（小数部分加0.5 强转整形下截断就得四舍五入结果）。
    //131072 = 1 << 17  1/s = (1+(std >>1))/std = 1/std + std/2/std = 1/std + 0.5  
    tmp32 = 131072 + (std >> 1);
    inv_std[0] = signalProcess.WebRtcSpl_DivW32W16(tmp32, std);

    // Calculate |inv_std2| = 1 / s^2, in Q14.
    tmp16[0] = (inv_std[0] >> 2);  // Q10 -> Q8.
    // Q-domain: (Q8 * Q8) >> 2 = Q14.
    inv_std2[0] = ((tmp16[0] * tmp16[0]) >> 2);
    // TODO(bjornv): Investigate if changing to
    // inv_std2 = (int16_t)((inv_std * inv_std) >> 6);
    // gives better accuracy.

    tmp16[0] = (input << 3);  // Q4 -> Q7
    tmp16[0] = tmp16[0] - mean;  // Q7 - Q7 = Q7

    // To be used later, when updating noise/speech model.
    // |delta| = (x - m) / s^2, in Q11.
    // Q-domain: (Q14 * Q7) >> 10 = Q11.
    delta[index] = ((inv_std2[0] * tmp16[0]) >> 10);

    // Calculate the exponent |tmp32| = (x - m)^2 / (2 * s^2), in Q10. Replacing
    // division by two with one shift.
    // Q-domain: (Q11 * Q7) >> 8 = Q10.
    tmp32 = (delta[index] * tmp16[0]) >> 9;

    // If the exponent is small enough to give a non-zero probability we calculate
    // |exp_value| ~= exp(-(x - m)^2 / (2 * s^2))
    //             ~= exp2(-log2(exp(1)) * |tmp32|).    (exp2(3) = 2^3)
    //             ~= 2^(-log2(exp(1)) * |tmp32|)
    //2^3=8 <=> 3=Math.log2(8) = Math.log2(2^3)=3*Math.log2(2)=3*1
    //2^3=8 = 2^Math.log2(8) = 8
    //exp(-(x - m)^2 / (2 * s^2)) = e^(-(x - m)^2 / (2 * s^2))
    //2^Math.log2(e^(-(x - m)^2 / (2 * s^2))) = e^(-(x - m)^2 / (2 * s^2))
    //2^Math.log2(e)*(-(x - m)^2 / (2 * s^2)) =  2^(-log2(exp(1)) * |tmp32|)
    //
    if (tmp32 < kCompVar) {
        // Calculate |tmp16| = log2(exp(1)) * |tmp32|, in Q10.
        // Q-domain: (Q12 * Q10) >> 12 = Q10.
        // tmp16[0]是一个负的Q10值。
        /*
        将tmp16 与 0x03FF按位与操作，取低10位，即Q10的小数部分记 f，由于tmp16是个负值，机器码为反码表示，那么得与的结果记为 f2=1−f, 比如tmp16为-3.6（Q0）, 那么tmp16 & 0x03FF为0.4（Q0）。那么再或上0x0400,exp_value在Q0就是一个1.x的值:exp_value = 1+f2
        接着，tmp16 异或0xFFFF, 即求反码，由于tmp16 是个负值，机器码求反码就变其绝对值（这里应该是求补码，即反码再加1，但这里省略近似，因为Q10的1 很是很小值）。接下来tmp16 >>= 10转为Q0的实数部分记为 r, 那么原来
        2^(-log2(exp(1)) * |tmp32|) = 2^(-c) c为实数部分r和小数f组成的值,即c=r+f
        2^(-c) = 2^-(r+f) = 2^-(r+1)*2^(1-f) = 2^-(r+1)*2^f2
        r+1便是tmp16 += 1这时候的tmp16值，即 tmp16=r+1,
        这里利用近似解：
        2^f2 ~= 1+f2
        exp_value >>= tmp16 = exp_value*2^(tmp16) = (1+f2)*2^(r+1)
        */
        tmp16[0] = ((kLog2Exp * tmp32) >> 12);
        tmp16[0] = -tmp16[0];
        exp_value[0] = (0x0400 | (tmp16[0] & 0x03FF));
        tmp16[0] ^= 0xFFFF;
        tmp16[0] >>= 10;
        tmp16[0] += 1;
        // Get |exp_value| = exp(-|tmp32|) in Q10.
        exp_value[0] >>= tmp16[0];
    }

    // Calculate and return (1 / s) * exp(-(x - m)^2 / (2 * s^2)), in Q20.
    // Q-domain: Q10 * Q10 = Q20.
    return inv_std[0] * exp_value[0];
}

module.exports = {
    WebRtcVad_GaussianProbability
}
