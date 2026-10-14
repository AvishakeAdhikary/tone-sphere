// Engine-level tests: graph mixing/gain, effects, encoding helpers.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "core/graph.h"
#include "control/remote.h"
#include "core/latency.h"
#include "core/presets.h"
#include "dsp/fx.h"
#include "util/encoding.h"
#include "util/platform.h"

using namespace ts;
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

#include "net/vban.h"

TEST_CASE("VBAN header matches the protocol specification", "[vban]") {
    vban::Header h;
    h.sampleRate = 48000;
    h.samples = 256;
    h.channels = 2;
    h.type = vban::DataType::Int16;
    h.stream = "Stream1";
    h.frame = 0x01020304;
    std::vector<float> in(512, 0.5f);
    uint8_t pkt[2048];
    const size_t len = vban::encode(h, in.data(), pkt, sizeof(pkt));
    REQUIRE(len == 28 + 256 * 2 * 2);
    REQUIRE(std::memcmp(pkt, "VBAN", 4) == 0);
    REQUIRE(pkt[4] == 3);    // 48 kHz index, audio sub-protocol
    REQUIRE(pkt[5] == 255);  // samples - 1
    REQUIRE(pkt[6] == 1);    // channels - 1
    REQUIRE(pkt[7] == 1);    // INT16, PCM codec
    REQUIRE(std::memcmp(pkt + 8, "Stream1\0\0\0\0\0\0\0\0\0", 16) == 0);
    REQUIRE(pkt[24] == 0x04);
    REQUIRE(pkt[27] == 0x01);
}

TEST_CASE("VBAN encode/decode round trip for every PCM type", "[vban]") {
    for (auto t : {vban::DataType::Int16, vban::DataType::Int24, vban::DataType::Int32, vban::DataType::Float32}) {
        vban::Header h;
        h.sampleRate = 44100;
        h.channels = 2;
        h.type = t;
        h.samples = vban::framesPerPacket(2, t);
        h.stream = "rt";
        std::vector<float> in(static_cast<size_t>(h.samples) * 2), out(in.size());
        for (size_t i = 0; i < in.size(); ++i) in[i] = std::sin(static_cast<float>(i) * 0.1f) * 0.8f;
        uint8_t pkt[2048];
        const size_t len = vban::encode(h, in.data(), pkt, sizeof(pkt));
        REQUIRE(len <= 28 + 1436);
        vban::Header p;
        const uint8_t* payload = nullptr;
        size_t pbytes = 0;
        REQUIRE(vban::parse(pkt, len, p, payload, pbytes));
        REQUIRE(p.sampleRate == 44100);
        REQUIRE(p.samples == h.samples);
        REQUIRE(p.stream == "rt");
        vban::decode(p, payload, out.data(), 2);
        for (size_t i = 0; i < in.size(); ++i) REQUIRE_THAT(out[i], WithinAbs(in[i], t == vban::DataType::Int16 ? 1e-4 : 1e-6));
    }
}

#ifdef _WIN32
#include <windows.h>
#include "backends/win/hotkeys.h"
TEST_CASE("Shortcut accelerators parse like the UI writes them", "[hotkeys]") {
    unsigned vk;
    bool c, a, s, w;
    REQUIRE(Hotkeys::parse("Ctrl+Shift+M", vk, c, a, s, w));
    REQUIRE((vk == 'M' && c && s && !a && !w));
    REQUIRE(Hotkeys::parse("Alt+F12", vk, c, a, s, w));
    REQUIRE((vk == VK_F12 && a && !c));
    REQUIRE(Hotkeys::parse("Ctrl+Space", vk, c, a, s, w));
    REQUIRE(vk == VK_SPACE);
    REQUIRE(Hotkeys::parse("Win+ArrowUp", vk, c, a, s, w));
    REQUIRE((vk == VK_UP && w));
    REQUIRE(Hotkeys::parse("Ctrl++", vk, c, a, s, w));
    REQUIRE(vk == VK_OEM_PLUS);
    REQUIRE_FALSE(Hotkeys::parse("Ctrl+Shift", vk, c, a, s, w));
}
#endif

TEST_CASE("Base64 round-trips binary plugin state", "[util]") {
    std::string bin;
    for (int i = 0; i < 1000; ++i) bin += static_cast<char>((i * 37 + 11) & 0xFF);
    for (size_t n : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(999), size_t(1000)}) {
        const std::string part = bin.substr(0, n);
        REQUIRE(ts::base64Decode(ts::base64Encode(part)) == part);
    }
    REQUIRE(ts::base64Decode("VG9uZVNwaGVyZQ==") == "ToneSphere");
}

TEST_CASE("Latency estimate finds a delayed, noisy chirp", "[latency]") {
    const double sr = 48000;
    const auto chirp = ts::makeLatencyChirp(sr);
    const size_t emitAt = 4800;
    std::vector<float> cap(52800, 0.0f);
    uint32_t seed = 1;
    for (auto& v : cap) {
        seed = seed * 1664525u + 1013904223u;
        v = 0.02f * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f);  // noise
    }
    for (size_t k = 0; k < chirp.size(); ++k) cap[emitAt + 1234 + k] += 0.3f * chirp[k];
    const auto e = ts::estimateLatency(cap, chirp, emitAt);
    REQUIRE_THAT(e.samples, Catch::Matchers::WithinAbs(1234.0, 0.5));
    REQUIRE(e.confidence > 0.8);

    // Half-sample delay (linear interpolation of the chirp): refined below one sample.
    std::vector<float> cap2(52800, 0.0f);
    for (size_t k = 0; k + 1 < chirp.size(); ++k) cap2[emitAt + 500 + k + 1] += 0.5f * (chirp[k] + chirp[k + 1]);
    const auto e2 = ts::estimateLatency(cap2, chirp, emitAt);
    REQUIRE_THAT(e2.samples, Catch::Matchers::WithinAbs(500.5, 0.3));

    // Nothing came back: low confidence.
    std::vector<float> silence(52800, 0.0f);
    for (auto& v : silence) {
        seed = seed * 1664525u + 1013904223u;
        v = 0.02f * (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f);
    }
    REQUIRE(ts::estimateLatency(silence, chirp, emitAt).confidence < 0.3);
}
namespace {
// Minimal OSC encoder for the tests.
void oscStr(std::vector<uint8_t>& b, const std::string& s) {
    b.insert(b.end(), s.begin(), s.end());
    b.push_back(0);
    while (b.size() % 4) b.push_back(0);
}
void oscF(std::vector<uint8_t>& b, float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    for (int s = 24; s >= 0; s -= 8) b.push_back(static_cast<uint8_t>(u >> s));
}
}  // namespace

TEST_CASE("OSC messages and bundles map to remote actions", "[remote]") {
    std::vector<uint8_t> m;
    oscStr(m, "/tonesphere/volume");
    oscStr(m, ",sf");
    oscStr(m, "Stream mix");
    oscF(m, 0.5f);
    std::vector<std::pair<std::string, std::vector<ts::OscArg>>> msgs;
    REQUIRE(ts::parseOscPacket(m.data(), m.size(), msgs));
    REQUIRE(msgs.size() == 1);
    auto a = ts::oscToAction(msgs[0].first, msgs[0].second);
    REQUIRE(a);
    REQUIRE(a->action == "volume");
    REQUIRE(a->target == "Stream mix");
    REQUIRE_THAT(*a->value, Catch::Matchers::WithinAbs(0.5, 1e-6));
    REQUIRE_THAT(ts::faderToDb(0.5f), Catch::Matchers::WithinAbs(-27.0, 1e-4));
    REQUIRE(ts::faderToDb(0.0f) == -90.0f);
    REQUIRE_THAT(ts::faderToDb(1.0f), Catch::Matchers::WithinAbs(6.0, 1e-4));

    // Bundle with a toggle (no value) and a scene change.
    std::vector<uint8_t> e1, e2, b;
    oscStr(e1, "/tonesphere/mute");
    oscStr(e1, ",s");
    oscStr(e1, "Microphone");
    oscStr(e2, "/tonesphere/scene");
    oscStr(e2, ",s");
    oscStr(e2, "Gaming");
    oscStr(b, "#bundle");
    for (int i = 0; i < 8; ++i) b.push_back(i == 7 ? 1 : 0);
    for (auto* e : {&e1, &e2}) {
        const uint32_t n = static_cast<uint32_t>(e->size());
        for (int s = 24; s >= 0; s -= 8) b.push_back(static_cast<uint8_t>(n >> s));
        b.insert(b.end(), e->begin(), e->end());
    }
    msgs.clear();
    REQUIRE(ts::parseOscPacket(b.data(), b.size(), msgs));
    REQUIRE(msgs.size() == 2);
    auto mute = ts::oscToAction(msgs[0].first, msgs[0].second);
    REQUIRE((mute && mute->action == "mute" && !mute->value));
    auto scene = ts::oscToAction(msgs[1].first, msgs[1].second);
    REQUIRE((scene && scene->action == "scene" && scene->target == "Gaming"));

    // Garbage and foreign addresses are ignored.
    const uint8_t junk[] = {1, 2, 3};
    msgs.clear();
    REQUIRE_FALSE(ts::parseOscPacket(junk, sizeof(junk), msgs));
    REQUIRE_FALSE(ts::oscToAction("/other/volume", {}));
}

TEST_CASE("MIDI controller messages map to remote actions", "[remote]") {
    const nlohmann::json maps = nlohmann::json::parse(R"([
        {"type":"cc","channel":1,"number":7,"action":"volume","target":"bus_1"},
        {"type":"cc","channel":0,"number":20,"action":"mute","target":"Microphone"},
        {"type":"note","channel":0,"number":60,"action":"solo","target":"Music"},
        {"type":"cc","channel":0,"number":30,"action":"scene","target":"Gaming"}
    ])");
    auto v = ts::midiToAction(maps, 0xB0, 7, 127);  // CC7 on channel 1
    REQUIRE((v && v->action == "volume" && *v->value == 1.0f));
    REQUIRE_FALSE(ts::midiToAction(maps, 0xB1, 7, 127));  // channel 2: not mapped
    auto m = ts::midiToAction(maps, 0xB5, 20, 10);         // any channel, switch "off"
    REQUIRE((m && m->action == "mute" && *m->value == 0.0f));
    auto s = ts::midiToAction(maps, 0x90, 60, 100);        // note on toggles
    REQUIRE((s && s->action == "solo" && !s->value));
    REQUIRE_FALSE(ts::midiToAction(maps, 0x90, 60, 0));     // note on with velocity 0 = note off
    REQUIRE_FALSE(ts::midiToAction(maps, 0xB0, 30, 0));     // scene triggers only on press
    REQUIRE(ts::midiToAction(maps, 0xB0, 30, 127));
}
TEST_CASE("Preset device rules match devices by bin, kind and name", "[presets]") {
    const auto rule = nlohmann::json::parse(R"json({"bin":"physical-in","kind":"wasapi","match":["Line (AI-04)","AI-04"]})json");
    // Windows names endpoints "Line (AI-04)": ToneSphere shows name "Line", subtitle "AI-04".
    const int exact = ts::presetMatchScore(rule, "physical-in", "wasapi", "Line", "AI-04");
    const int loose = ts::presetMatchScore(rule, "physical-in", "wasapi", "Microphone", "AI-04 USB");
    REQUIRE(exact > loose);
    REQUIRE(loose > 0);
    REQUIRE(ts::presetMatchScore(rule, "physical-out", "wasapi", "Line", "AI-04") == 0);  // wrong direction
    REQUIRE(ts::presetMatchScore(rule, "physical-in", "asio", "Line", "AI-04") == 0);     // wrong kind
    REQUIRE(ts::presetMatchScore(rule, "physical-in", "wasapi", "Microphone Array", "Intel Smart Sound") == 0);
    REQUIRE(ts::presetMatchScore(rule, "physical-in", "wasapi", "LINE", "ai-04") == exact);  // case-insensitive
    // Earlier patterns win over later ones.
    const auto two = nlohmann::json::parse(R"json({"bin":"physical-out","match":["Speakers (Realtek(R) Audio)","Speakers"]})json");
    REQUIRE(ts::presetMatchScore(two, "physical-out", "wasapi", "Speakers", "Realtek(R) Audio") >
            ts::presetMatchScore(two, "physical-out", "wasapi", "Speakers", "AI-04"));
    REQUIRE(ts::deviceLabel("Line", "AI-04") == "Line (AI-04)");
    REQUIRE(ts::deviceLabel("Stream mix", "") == "Stream mix");
}
TEST_CASE("OSC addresses from before the rename (/nextone/) still work", "[remote]") {
    std::vector<ts::OscArg> args(1);
    args[0].type = 's';
    args[0].str = "Gaming";
    auto a = ts::oscToAction("/nextone/scene", args);
    REQUIRE(a);
    REQUIRE(a->action == "scene");
    REQUIRE(a->target == "Gaming");
    REQUIRE_FALSE(ts::oscToAction("/other/scene", args));
    REQUIRE_FALSE(ts::oscToAction("/tonesphere/", args));
}

TEST_CASE("Settings from the NexTone data folder migrate once into a fresh ToneSphere folder", "[platform]") {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path() / ("ts-migrate-" + ts::randomHex(4));
    const fs::path from = root / "NexTone", to = root / "ToneSphere";
    fs::create_directories(from / "presets");
    auto write = [](const fs::path& p, const char* s) { std::ofstream(p, std::ios::binary) << s; };
    write(from / "config.json", "{\"old\":1}");
    write(from / "plugins.json", "[]");
    write(from / "engine.json", "{}");  // the old engine's discovery file: never copied
    write(from / "engine.log", "log");
    write(from / "presets" / "a.json", "{}");

    REQUIRE(ts::migrateLegacyData(from.string(), to.string()) == 3);
    REQUIRE(fs::exists(to / "config.json"));
    REQUIRE(fs::exists(to / "plugins.json"));
    REQUIRE(fs::exists(to / "presets" / "a.json"));
    REQUIRE_FALSE(fs::exists(to / "engine.json"));
    REQUIRE_FALSE(fs::exists(to / "engine.log"));

    // Once ToneSphere has its own config, nothing is copied again (or overwritten).
    write(to / "config.json", "{\"new\":1}");
    write(from / "extra.json", "{}");
    REQUIRE(ts::migrateLegacyData(from.string(), to.string()) == 0);
    REQUIRE_FALSE(fs::exists(to / "extra.json"));
    // A missing legacy folder is not an error.
    REQUIRE(ts::migrateLegacyData((root / "missing").string(), (root / "x").string()) == 0);
    fs::remove_all(root);
}
