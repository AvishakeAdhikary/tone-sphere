// Unit tests for the real-time core: ring buffer, resampler, drift compensation.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>
#include <thread>
#include <vector>

#include "core/bridge.h"
#include "core/resampler.h"
#include "core/ring_buffer.h"

using namespace ts;
using Catch::Matchers::WithinAbs;

static constexpr double kPi = 3.14159265358979323846;

TEST_CASE("FrameRing basic read/write and wraparound", "[ring]") {
    FrameRing r;
    r.init(2, 8);
    REQUIRE(r.capacity() == 8);
    std::vector<float> in(12), out(16);  // up to 8 stereo frames
    for (int i = 0; i < 12; ++i) in[i] = float(i);
    REQUIRE(r.write(in.data(), 6) == 6);
    REQUIRE(r.available() == 6);
    REQUIRE(r.read(out.data(), 4) == 4);
    for (int i = 0; i < 8; ++i) REQUIRE(out[i] == float(i));
    REQUIRE(r.write(in.data(), 6) == 6);  // wraps
    REQUIRE(r.available() == 8);
    REQUIRE(r.space() == 0);
    REQUIRE(r.write(in.data(), 1) == 0);
    REQUIRE(r.read(out.data(), 8) == 8);
    REQUIRE(out[0] == 8.0f);
    REQUIRE(out[4] == 0.0f);
}

TEST_CASE("FrameRing SPSC stress keeps order", "[ring]") {
    FrameRing r;
    r.init(1, 256);
    constexpr int N = 2'000'000;
    std::thread producer([&] {
        int i = 0;
        float buf[37];
        while (i < N) {
            int n = 0;
            for (; n < 37 && i + n < N; ++n) buf[n] = float(i + n);
            size_t w = 0;
            while (w < size_t(n)) w += r.write(buf + w, n - w);
            i += n;
        }
    });
    int expect = 0;
    bool ok = true;
    float buf[53];
    while (expect < N) {
        size_t got = r.read(buf, 53);
        for (size_t k = 0; k < got; ++k) ok &= (buf[k] == float(expect++));
    }
    producer.join();
    REQUIRE(ok);
}

static double sineAmplitudeError(const std::vector<float>& x, double freq, double rate, size_t skip) {
    // Least-squares fit of a sine at known frequency; returns RMS residual relative to amplitude.
    double ss = 0, sc = 0, cc = 0, xs = 0, xc = 0;
    for (size_t i = skip; i < x.size(); ++i) {
        double s = std::sin(2 * kPi * freq * double(i) / rate), c = std::cos(2 * kPi * freq * double(i) / rate);
        ss += s * s; sc += s * c; cc += c * c; xs += x[i] * s; xc += x[i] * c;
    }
    double det = ss * cc - sc * sc;
    double a = (xs * cc - xc * sc) / det, b = (xc * ss - xs * sc) / det;
    double res = 0;
    for (size_t i = skip; i < x.size(); ++i) {
        double s = std::sin(2 * kPi * freq * double(i) / rate), c = std::cos(2 * kPi * freq * double(i) / rate);
        double e = x[i] - (a * s + b * c);
        res += e * e;
    }
    double amp = std::sqrt(a * a + b * b);
    return std::sqrt(res / double(x.size() - skip)) / amp;
}

TEST_CASE("Resampler converts 44.1k -> 48k with low distortion", "[resampler]") {
    const double inRate = 44100, outRate = 48000, f = 1000;
    VarispeedResampler rs;
    rs.init(1, 4096, 0.95);
    std::vector<float> in(44100);
    for (size_t i = 0; i < in.size(); ++i) in[i] = float(0.5 * std::sin(2 * kPi * f * double(i) / inRate));
    std::vector<float> out;
    std::vector<float> blk(512);
    float* o[1] = {blk.data()};
    size_t pos = 0;
    const double step = inRate / outRate;
    while (pos + 512 < in.size()) {
        size_t need = rs.inputNeeded(256, step);
        const float* p[1] = {in.data() + pos};
        pos += rs.push(p, need);
        size_t got = rs.pull(o, 256, step);
        out.insert(out.end(), blk.begin(), blk.begin() + got);
    }
    REQUIRE(out.size() > 40000);
    // Compensate the filter group delay (16 input frames) by fitting phase freely.
    double err = sineAmplitudeError(out, f, outRate, 200);
    INFO("relative residual " << err);
    REQUIRE(err < 1e-3);  // better than -60 dB
}

TEST_CASE("Resampler unity ratio preserves signal", "[resampler]") {
    VarispeedResampler rs;
    rs.init(2, 1024);
    std::vector<float> l(1000), r(1000), ol(1000), orr(1000);
    for (int i = 0; i < 1000; ++i) { l[i] = float(std::sin(i * 0.05)); r[i] = -l[i]; }
    const float* in[2] = {l.data(), r.data()};
    rs.push(in, 1000);
    float* out[2] = {ol.data(), orr.data()};
    size_t got = rs.pull(out, 900, 1.0);
    REQUIRE(got == 900);
    // Output is delayed by halfTaps-1 frames relative to input.
    for (int i = 100; i < 800; ++i) REQUIRE_THAT(ol[i], WithinAbs(l[i], 2e-3));
    for (int i = 100; i < 800; ++i) REQUIRE_THAT(orr[i], WithinAbs(r[i], 2e-3));
}

// Simulates a capture device whose crystal runs `ppm` off, feeding the engine through
// a CaptureBridge; checks that the controller locks with no underruns/overruns.
static void simulateCapture(double ppm, double seconds, uint64_t& xruns, double& finalPpm, double devNominal = 48000, int devFrames = 480) {
    BridgeConfig cfg;
    cfg.channels = 2;
    cfg.deviceRate = 48000;
    cfg.engineRate = 48000;
    cfg.devicePeriod = 480;
    cfg.engineMaxBlock = 128;
    CaptureBridge b;
    b.init(cfg);
    b.markStarted();
    const double devRate = 48000 * (1 + ppm * 1e-6);
    std::vector<float> dev(480 * 2, 0.25f);
    std::vector<float> l(128), r(128);
    float* out[2] = {l.data(), r.data()};
    double tDev = 0, tEng = 0;
    const double devPeriod = devFrames / devRate, engPeriod = 128 / 48000.0;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> jitter(0, 0.0015);
    uint64_t warmupX = 0;
    double ppmSum = 0; long ppmN = 0;
    for (double t = 0; t < seconds; t += 0.0005) {
        while (tDev <= t) { b.deviceWrite(dev.data(), static_cast<size_t>(devFrames), tDev); tDev += devPeriod; }
        while (tEng + jitter(rng) <= t) { b.engineRead(out, 128, t); tEng += engPeriod; }
        if (t < 30) warmupX = b.stats().underruns + b.stats().overruns;
        if (t > seconds - 60) { ppmSum += b.stats().ppm; ++ppmN; }
    }
    xruns = b.stats().underruns + b.stats().overruns - warmupX;
    finalPpm = ppmSum / double(ppmN);
}

TEST_CASE("Capture bridge locks to a fast device clock (+300 ppm)", "[drift]") {
    uint64_t x; double p;
    simulateCapture(300, 600, x, p);
    INFO("ppm " << p << " xruns after warmup " << x);
    REQUIRE(x == 0);
    REQUIRE_THAT(p, WithinAbs(300, 15));
}

TEST_CASE("Capture bridge locks to a slow device clock (-250 ppm)", "[drift]") {
    uint64_t x; double p;
    simulateCapture(-250, 600, x, p);
    INFO("ppm " << p << " xruns after warmup " << x);
    REQUIRE(x == 0);
    REQUIRE_THAT(p, WithinAbs(-250, 15));
}

TEST_CASE("Capture bridge locks a 44.1 kHz device into a 48 kHz engine", "[drift]") {
    // A USB interface whose Windows format is 44.1 kHz, delivering 10 ms packets (441 frames).
    uint64_t x; double p;
    simulateCapture(120, 600, x, p, 44100, 441);
    INFO("ppm " << p << " xruns after warmup " << x);
    REQUIRE(x == 0);
    REQUIRE_THAT(p, WithinAbs(120, 15));
}

TEST_CASE("Render bridge locks and does not underrun", "[drift]") {
    BridgeConfig cfg;
    cfg.channels = 2;
    cfg.deviceRate = 44100;  // also exercises real SRC
    cfg.engineRate = 48000;
    cfg.devicePeriod = 441;
    cfg.engineMaxBlock = 256;
    RenderBridge b;
    b.init(cfg);
    b.markStarted();
    const double devRate = 44100 * (1 - 180e-6);
    std::vector<float> dev(441 * 2);
    std::vector<float> l(256, 0.1f), r(256, 0.1f);
    const float* in[2] = {l.data(), r.data()};
    double tDev = 0, tEng = 0;
    uint64_t warm = 0;
    double rp = 0; long rn = 0;
    for (double t = 0; t < 600; t += 0.0005) {
        while (tEng <= t) { b.engineWrite(in, 256, t); tEng += 256 / 48000.0; }
        while (tDev <= t) { b.deviceRead(dev.data(), 441, tDev); tDev += 441 / devRate; }
        if (t < 30) warm = b.stats().underruns + b.stats().overruns;
        if (t > 540) { rp += b.stats().ppm; ++rn; }
    }
    uint64_t x = b.stats().underruns + b.stats().overruns - warm;
    INFO("avg ppm " << rp / rn << " xruns " << x);
    REQUIRE(x == 0);
    // Device 180 ppm slow -> engine must produce ~180 ppm fewer device frames (positive correction).
    REQUIRE_THAT(rp / double(rn), WithinAbs(180, 15));
}

TEST_CASE("Capture bridge follows a device that runs 1 % slow (cheap USB interface)", "[drift]") {
    // Measured on a real AI-04 at 44.1 kHz: about -10,300 ppm. Beyond the PI loop's own range:
    // the measured device rate (feed-forward) has to carry it.
    uint64_t x; double p;
    simulateCapture(-10300, 600, x, p, 44100, 441);
    INFO("ppm " << p << " xruns after warmup " << x);
    REQUIRE(x == 0);
    REQUIRE_THAT(p, WithinAbs(-10300, 60));
}

TEST_CASE("Render bridge follows a device that runs 1 % slow", "[drift]") {
    BridgeConfig cfg;
    cfg.channels = 2;
    cfg.deviceRate = 44100;
    cfg.engineRate = 48000;
    cfg.devicePeriod = 441;
    cfg.engineMaxBlock = 128;
    RenderBridge b;
    b.init(cfg);
    b.markStarted();
    const double devRate = 44100 * (1 - 0.0100);
    std::vector<float> dev(441 * 2);
    std::vector<float> l(128, 0.1f), r(128, 0.1f);
    const float* in[2] = {l.data(), r.data()};
    double tDev = 0, tEng = 0;
    uint64_t warm = 0;
    double rp = 0; long rn = 0;
    for (double t = 0; t < 600; t += 0.0005) {
        while (tEng <= t) { b.engineWrite(in, 128, t); tEng += 128 / 48000.0; }
        while (tDev <= t) { b.deviceRead(dev.data(), 441, tDev); tDev += 441 / devRate; }
        if (t < 30) warm = b.stats().underruns + b.stats().overruns;
        if (t > 540) { rp += b.stats().ppm; ++rn; }
    }
    const uint64_t x = b.stats().underruns + b.stats().overruns - warm;
    const double ppm = rp / double(rn);
    INFO("ppm " << ppm << " xruns after warmup " << x);
    REQUIRE(x == 0);
    REQUIRE_THAT(ppm, WithinAbs(1e6 * (1 / 0.99 - 1), 60));  // a slower device needs +ppm
}