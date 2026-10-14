#pragma once
// RBJ "Audio EQ Cookbook" biquads, transposed direct form II.
#include <cmath>

namespace nt {

struct Biquad {
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    double z1[8] = {}, z2[8] = {};

    void reset() {
        for (int i = 0; i < 8; ++i) z1[i] = z2[i] = 0;
    }
    inline float tick(int ch, float x) {
        const double y = b0 * x + z1[ch];
        z1[ch] = b1 * x - a1 * y + z2[ch];
        z2[ch] = b2 * x - a2 * y;
        return static_cast<float>(y);
    }

    void setPeaking(double sr, double f, double q, double gainDb) {
        const double A = std::pow(10.0, gainDb / 40.0), w = 2 * kPi * f / sr, al = std::sin(w) / (2 * q), c = std::cos(w);
        norm(1 + al * A, -2 * c, 1 - al * A, 1 + al / A, -2 * c, 1 - al / A);
    }
    void setLowShelf(double sr, double f, double gainDb) {
        const double A = std::pow(10.0, gainDb / 40.0), w = 2 * kPi * f / sr, c = std::cos(w), s = std::sin(w);
        const double al = s / 2 * std::sqrt(2.0), sq = 2 * std::sqrt(A) * al;
        norm(A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq),
             (A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq);
    }
    void setHighShelf(double sr, double f, double gainDb) {
        const double A = std::pow(10.0, gainDb / 40.0), w = 2 * kPi * f / sr, c = std::cos(w), s = std::sin(w);
        const double al = s / 2 * std::sqrt(2.0), sq = 2 * std::sqrt(A) * al;
        norm(A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq),
             (A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq);
    }
    void setHighPass(double sr, double f, double q = 0.7071) {
        const double w = 2 * kPi * f / sr, al = std::sin(w) / (2 * q), c = std::cos(w);
        norm((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al);
    }

private:
    static constexpr double kPi = 3.14159265358979323846;
    void norm(double nb0, double nb1, double nb2, double na0, double na1, double na2) {
        b0 = nb0 / na0;
        b1 = nb1 / na0;
        b2 = nb2 / na0;
        a1 = na1 / na0;
        a2 = na2 / na0;
    }
};

}  // namespace nt
