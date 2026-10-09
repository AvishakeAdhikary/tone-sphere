// Engine-level tests: graph mixing/gain, effects, encoding helpers.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "core/graph.h"
#include "dsp/fx.h"
#include "util/encoding.h"

using namespace nt;
using Catch::Matchers::WithinAbs;

namespace {
std::shared_ptr<NodeRuntime> makeNode(const char* id, Bin bin, int ch) {
    auto n = std::make_shared<NodeRuntime>();
    n->id = id;
    n->bin = bin;
    n->allocate(ch, 48000);
    return n;
}
}  // namespace

TEST_CASE("Tone -> bus -> output mixes with route and crosspoint gains", "[graph]") {
    auto tone = makeNode("tone", Bin::PhysicalIn, 2);
    tone->tone = std::make_shared<ToneGen>();
    tone->tone->enabled = true;
    tone->tone->freq = 1000;
    tone->tone->levelDb = -6.0206f;  // amplitude 0.5
    auto bus = makeNode("bus", Bin::Bus, 2);
    auto out = makeNode("out", Bin::PhysicalOut, 2);
    auto r1 = std::make_shared<RouteRuntime>();
    auto r2 = std::make_shared<RouteRuntime>();
    r2->gainDb = -6.0206f;  // x0.5

    Program p;
    p.sampleRate = 48000;
    p.meters = std::make_shared<MeterBank>(12, 1);
    ProgramNode a{tone.get(), true, false, false, false, {}, {}, 0};
    ProgramNode b{bus.get(), true, true, false, false, {}, {}, 4};
    b.in.push_back({tone.get(), r1.get(), {{0, 0, 1.0f}, {1, 1, 1.0f}}});
    ProgramNode c{out.get(), false, false, true, true, {}, {}, 8};
    c.in.push_back({bus.get(), r2.get(), {{0, 0, 1.0f}}});  // left only
    p.nodes = {a, b, c};

    // Gains fade in from silence and meters hold peaks briefly: run long enough to settle.
    for (int i = 0; i < 60; ++i) processProgram(p, 256, 0.0);
    float peakL = 0, peakR = 0;
    for (int i = 0; i < 256; ++i) {
        peakL = std::max(peakL, std::fabs(out->buf[0][i]));
        peakR = std::max(peakR, std::fabs(out->buf[1][i]));
    }
    REQUIRE_THAT(peakL, WithinAbs(0.25, 0.01));
    REQUIRE(peakR == 0.0f);
    // Meters were written for the output (offset 8 -> L peak).
    REQUIRE_THAT(p.meters->values[8].load(), WithinAbs(0.25, 0.02));
}

TEST_CASE("Muting a node ramps to silence without clicks", "[graph]") {
    auto tone = makeNode("tone", Bin::PhysicalIn, 1);
    tone->tone = std::make_shared<ToneGen>();
    tone->tone->enabled = true;
    tone->tone->levelDb = 0;
    Program p;
    p.sampleRate = 48000;
    p.nodes = {ProgramNode{tone.get(), true, false, false, false, {}, {}, -1}};
    processProgram(p, 128, 0);
    tone->muted = true;
    processProgram(p, 128, 0);
    // Ramp: the last samples of the muting block are near zero, the first are not forced to zero.
    REQUIRE(std::fabs(tone->buf[0][127]) < 0.02f);
    processProgram(p, 128, 0);
    for (int i = 0; i < 128; ++i) REQUIRE(tone->buf[0][i] == 0.0f);
}

TEST_CASE("Limiter keeps peaks under the ceiling", "[fx]") {
    auto fx = createBuiltinFx("limiter");
    REQUIRE(fx);
    fx->setParam("ceiling", -6.0f);
    fx->prepare(48000, 512, 1);
    std::vector<float> x(512);
    float* buf[1] = {x.data()};
    float maxOut = 0;
    for (int blk = 0; blk < 20; ++blk) {
        for (int i = 0; i < 512; ++i) x[i] = 0.95f * std::sin(i * 0.05f);
        fx->process(buf, 1, 512);
        if (blk > 1)
            for (float v : x) maxOut = std::max(maxOut, std::fabs(v));
    }
    REQUIRE(maxOut <= std::pow(10.0f, -6.0f / 20.0f) + 1e-4f);
}

TEST_CASE("Noise gate closes on quiet input", "[fx]") {
    auto fx = createBuiltinFx("gate");
    fx->setParam("threshold", -30.0f);
    fx->setParam("range", -80.0f);
    fx->prepare(48000, 1024, 1);
    std::vector<float> x(1024, 0.001f);  // -60 dB
    float* buf[1] = {x.data()};
    for (int i = 0; i < 40; ++i) {
        std::fill(x.begin(), x.end(), 0.001f);
        fx->process(buf, 1, 1024);
    }
    REQUIRE(std::fabs(x.back()) < 0.0001f);
}

TEST_CASE("Compressor reduces loud signals", "[fx]") {
    auto fx = createBuiltinFx("compressor");
    fx->setParam("threshold", -20.0f);
    fx->setParam("ratio", 4.0f);
    fx->setParam("knee", 0.0f);
    fx->prepare(48000, 1024, 1);
    std::vector<float> x(1024);
    float* buf[1] = {x.data()};
    for (int i = 0; i < 50; ++i) {
        std::fill(x.begin(), x.end(), 1.0f);  // 0 dBFS DC: 20 dB over -> 15 dB reduction
        fx->process(buf, 1, 1024);
    }
    REQUIRE_THAT(20 * std::log10(x.back()), WithinAbs(-15.0, 0.5));
}

TEST_CASE("Every built-in effect can be created and run", "[fx]") {
    for (const char* t : {"gate", "denoise", "eq", "compressor", "limiter", "ducker", "delay"}) {
        auto fx = createBuiltinFx(t);
        REQUIRE(fx);
        fx->prepare(48000, 480, 2);
        std::vector<float> l(480, 0.1f), r(480, -0.1f);
        float* buf[2] = {l.data(), r.data()};
        for (int i = 0; i < 4; ++i) fx->process(buf, 2, 480);
        for (float v : l) REQUIRE(std::isfinite(v));
    }
}

TEST_CASE("SHA-1 and base64 match the WebSocket RFC example", "[encoding]") {
    // RFC 6455 section 1.3
    const auto d = sha1("dGhlIHNhbXBsZSBub25jZQ==258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    REQUIRE(base64Encode(d.data(), d.size()) == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    REQUIRE(base64Encode(std::string_view("hello")) == "aGVsbG8=");
}
