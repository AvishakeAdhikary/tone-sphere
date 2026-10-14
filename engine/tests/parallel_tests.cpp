// Multi-core DSP: the parallel schedule must compute exactly what the serial order computes,
// and the worker pool must run every item exactly once, block after block.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "core/dsp_pool.h"
#include "core/graph.h"
#include "dsp/fx.h"

using namespace ts;

namespace {

// A deterministic effect that also costs real time, like a plugin would.
class BusyFx : public Fx {
public:
    explicit BusyFx(double micros) : Fx("busy", "Busy", {}), micros_(micros) {}
    void process(float* const* buf, int channels, int frames) override {
        const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double, std::micro>(micros_);
        while (std::chrono::steady_clock::now() < until) {
        }
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < frames; ++i) buf[c][i] = buf[c][i] * 0.9f + 0.05f * std::sin(state_ += 0.01f);
    }

private:
    double micros_;
    float state_ = 0;
};

struct Rig {
    std::vector<std::shared_ptr<NodeRuntime>> nodes;
    std::vector<std::shared_ptr<RouteRuntime>> routes;
    Program p;

    NodeRuntime* add(const char* id, Bin bin, int ch, bool src, bool bus, bool dest, bool phys, std::vector<std::shared_ptr<Fx>> fx = {}) {
        auto n = std::make_shared<NodeRuntime>();
        n->id = id;
        n->bin = bin;
        n->allocate(ch, 48000);
        nodes.push_back(n);
        ProgramNode pn;
        pn.rt = n.get();
        pn.isSource = src;
        pn.isBus = bus;
        pn.isDest = dest;
        pn.physicalOut = phys;
        pn.fx = std::move(fx);
        pn.meterOffset = static_cast<int>(p.nodes.size()) * 4;
        for (auto& f : pn.fx) f->prepare(48000, kMaxBlock, ch);
        p.nodes.push_back(std::move(pn));
        return n.get();
    }
    int indexOf(NodeRuntime* n) const {
        for (size_t i = 0; i < p.nodes.size(); ++i)
            if (p.nodes[i].rt == n) return static_cast<int>(i);
        return -1;
    }
    void route(NodeRuntime* from, NodeRuntime* to, float gainDb = 0) {
        auto r = std::make_shared<RouteRuntime>();
        r->gainDb = gainDb;
        routes.push_back(r);
        p.nodes[static_cast<size_t>(indexOf(to))].in.push_back({from, r.get(), {{0, 0, 1.0f}, {1, 1, 0.7f}, {0, 1, 0.2f}}});
    }
    NodeRuntime* tone(const char* id, float freq, float db) {
        NodeRuntime* n = add(id, Bin::PhysicalIn, 2, true, false, false, false);
        n->tone = std::make_shared<ToneGen>();
        n->tone->freq = freq;
        n->tone->levelDb = db;
        return n;
    }
};

// Sources -> mixes (with a feedback loop and sidechains both ways) -> outputs, plus a solo mix.
std::unique_ptr<Rig> makeRig(double busyMicros) {
    auto r = std::make_unique<Rig>();
    auto fx = [&](const char* type) {
        auto f = createBuiltinFx(type);
        return f;
    };
    NodeRuntime* a = r->tone("a", 220, -12);
    NodeRuntime* b = r->tone("b", 330, -9);
    NodeRuntime* c = r->tone("c", 440, -6);
    NodeRuntime* d = r->tone("d", 97, -3);
    // A source whose ducker listens to a source processed after it (reads its previous block).
    auto duckA = fx("ducker");
    r->p.nodes[static_cast<size_t>(r->indexOf(a))].fx.push_back(duckA);
    duckA->prepare(48000, kMaxBlock, 2);
    duckA->sidechain.store(d->buf);
    duckA->sidechainChannels.store(2);
    r->p.nodes[static_cast<size_t>(r->indexOf(a))].sidechainFrom.push_back(r->indexOf(d));

    NodeRuntime* m1 = r->add("m1", Bin::Bus, 2, true, true, false, false, {fx("eq"), fx("compressor"), std::make_shared<BusyFx>(busyMicros)});
    NodeRuntime* m2 = r->add("m2", Bin::Bus, 2, true, true, false, false, {fx("limiter"), std::make_shared<BusyFx>(busyMicros)});
    NodeRuntime* m3 = r->add("m3", Bin::Bus, 2, true, true, false, false, {fx("gate"), std::make_shared<BusyFx>(busyMicros)});
    NodeRuntime* m4 = r->add("m4", Bin::Bus, 2, true, true, false, false, {std::make_shared<BusyFx>(busyMicros)});
    r->route(a, m1);
    r->route(b, m1, -3);
    r->route(c, m2);
    r->route(d, m3);
    r->route(b, m4);
    r->route(m1, m2, -6);  // forward
    r->route(m2, m1, -20); // feedback: m1 hears m2's previous block
    r->route(m2, m3);
    // A mix ducked by another mix.
    auto duckM4 = fx("ducker");
    duckM4->prepare(48000, kMaxBlock, 2);
    duckM4->sidechain.store(m1->buf);
    duckM4->sidechainChannels.store(2);
    r->p.nodes[static_cast<size_t>(r->indexOf(m4))].fx.push_back(duckM4);
    r->p.nodes[static_cast<size_t>(r->indexOf(m4))].sidechainFrom.push_back(r->indexOf(m1));

    NodeRuntime* o1 = r->add("o1", Bin::PhysicalOut, 2, false, false, true, true, {std::make_shared<BusyFx>(busyMicros)});
    NodeRuntime* o2 = r->add("o2", Bin::PhysicalOut, 2, false, false, true, true, {std::make_shared<BusyFx>(busyMicros)});
    NodeRuntime* o3 = r->add("o3", Bin::VirtualIn, 2, false, false, true, false, {std::make_shared<BusyFx>(busyMicros)});
    NodeRuntime* o4 = r->add("o4", Bin::AppIn, 2, false, false, true, false, {std::make_shared<BusyFx>(busyMicros)});
    r->route(m1, o1);
    r->route(m2, o2);
    r->route(m3, o3);
    r->route(m4, o4);
    r->route(a, o4, -6);
    r->p.sampleRate = 48000;
    r->p.meters = std::make_shared<MeterBank>(r->p.nodes.size() * 4, 1);
    (void)o1;
    (void)o2;
    (void)o3;
    return r;
}

bool sameBuffers(const Rig& x, const Rig& y, int frames) {
    for (size_t i = 0; i < x.nodes.size(); ++i)
        for (int c = 0; c < x.nodes[i]->channels; ++c)
            if (std::memcmp(x.nodes[i]->buf[c], y.nodes[i]->buf[c], sizeof(float) * static_cast<size_t>(frames)) != 0) return false;
    return true;
}

}  // namespace

TEST_CASE("Levels follow routes, sidechains, feedback and the source/destination split", "[parallel]") {
    auto r = makeRig(0);
    computeLevels(r->p);
    std::vector<int> level(r->p.nodes.size(), -1);
    for (size_t k = 0; k + 1 < r->p.levelStart.size(); ++k)
        for (int i = r->p.levelStart[k]; i < r->p.levelStart[k + 1]; ++i) level[static_cast<size_t>(r->p.levelNodes[static_cast<size_t>(i)])] = static_cast<int>(k);
    for (int l : level) REQUIRE(l >= 0);  // every node scheduled exactly once
    REQUIRE(r->p.levelNodes.size() == r->p.nodes.size());
    REQUIRE(r->p.parallelizable);
    auto L = [&](const char* id) {
        for (size_t i = 0; i < r->p.nodes.size(); ++i)
            if (r->p.nodes[i].rt->id == id) return level[i];
        return -1;
    };
    REQUIRE(L("b") == 0);
    REQUIRE(L("d") > L("a"));    // a's ducker reads d's previous block: a must go first
    REQUIRE(L("m1") > L("a"));
    REQUIRE(L("m2") > L("m1"));  // forward route
    REQUIRE(L("m3") > L("m2"));
    REQUIRE(L("m4") > L("m1"));  // sidechain
    for (const char* o : {"o1", "o2", "o3", "o4"}) {
        REQUIRE(L(o) > L("d"));  // destinations after every source
    }
    REQUIRE(L("o2") > L("m2"));
}

TEST_CASE("Parallel processing is bit-identical to the serial order", "[parallel]") {
    auto serial = makeRig(0);
    auto parallel = makeRig(0);
    computeLevels(serial->p);
    computeLevels(parallel->p);
    DspPool pool(3);
    DspScheduler sched;
    sched.pool = &pool;
    // Every mode, switched while running: forced 4 and 2 cores, auto, one core.
    const int modes[] = {4, 2, 4, 0, 1, 3, 4};
    for (int block = 0; block < 400; ++block) {
        sched.mode.store(modes[(block / 7) % 7]);
        // Solo a mix halfway through (physical outputs then carry only it).
        if (block == 200) {
            serial->p.soloBuses = {serial->nodes[5].get()};
            parallel->p.soloBuses = {parallel->nodes[5].get()};
            computeLevels(serial->p);
            computeLevels(parallel->p);
        }
        const int frames = 64 + (block % 5) * 48;
        processProgram(serial->p, frames, 0.0);
        processProgram(parallel->p, frames, 0.0, &sched);
        REQUIRE(sameBuffers(*serial, *parallel, frames));
    }
}

TEST_CASE("Auto mode spreads a heavy graph over cores and keeps a light one on one", "[parallel]") {
    if (DspPool::defaultWorkers() < 1) SKIP("single-core machine");
    DspPool pool(std::min(3, DspPool::defaultWorkers()));
    {
        auto light = makeRig(0);
        computeLevels(light->p);
        DspScheduler sched;
        sched.pool = &pool;
        for (int i = 0; i < 200; ++i) processProgram(light->p, 128, 0.0, &sched);
        REQUIRE(sched.threadsInUse.load() == 1);
    }
    {
        // 8 nodes x 300 us in a 2.67 ms block: far more than a quarter of the period.
        auto heavy = makeRig(300);
        computeLevels(heavy->p);
        DspScheduler sched;
        sched.pool = &pool;
        for (int i = 0; i < 120; ++i) processProgram(heavy->p, 128, 0.0, &sched);
        REQUIRE(sched.threadsInUse.load() > 1);
        sched.mode.store(1);
        processProgram(heavy->p, 128, 0.0, &sched);
        REQUIRE(sched.threadsInUse.load() == 1);
    }
}

TEST_CASE("The DSP pool runs every item exactly once, block after block", "[parallel]") {
    DspPool pool(4);
    REQUIRE(pool.workers() == 4);
    struct Ctx {
        std::vector<std::atomic<int>> hits;
        explicit Ctx(size_t n) : hits(n) {}
    } ctx(64);
    auto fn = [](void* c, int i) { static_cast<Ctx*>(c)->hits[static_cast<size_t>(i)].fetch_add(1, std::memory_order_relaxed); };
    std::mt19937 rng(7);
    for (int round = 0; round < 20000; ++round) {
        const int count = 1 + static_cast<int>(rng() % 64);
        const int helpers = static_cast<int>(rng() % 5);
        for (auto& h : ctx.hits) h.store(0);
        pool.parallelFor(count, fn, &ctx, helpers);
        for (int i = 0; i < 64; ++i) REQUIRE(ctx.hits[static_cast<size_t>(i)].load() == (i < count ? 1 : 0));
        // Now and then a pause long enough for the workers to park, so wake-ups are exercised too.
        if (round % 2000 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
