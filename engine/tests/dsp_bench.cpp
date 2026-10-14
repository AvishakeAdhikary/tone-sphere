// Measures one audio block of a heavy graph on one core and on several (the multi-core speed-up
// recorded in docs/STATUS.md). 8 inputs, each with noise suppression + EQ + compressor, into
// 8 outputs. Prints microseconds per 128-frame block.
//   tonesphere-dsp-bench [inputs] [blocks]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "core/dsp_pool.h"
#include "core/graph.h"
#include "dsp/fx.h"

using namespace ts;

namespace {
struct Graph {
    std::vector<std::shared_ptr<NodeRuntime>> nodes;
    std::vector<std::shared_ptr<RouteRuntime>> routes;
    Program p;
};

std::unique_ptr<Graph> build(int inputs) {
    auto g = std::make_unique<Graph>();
    g->p.sampleRate = 48000;
    auto add = [&](Bin bin, bool src, bool dest, std::vector<std::shared_ptr<Fx>> fx) {
        auto n = std::make_shared<NodeRuntime>();
        n->bin = bin;
        n->allocate(2, 48000);
        g->nodes.push_back(n);
        ProgramNode pn;
        pn.rt = n.get();
        pn.isSource = src;
        pn.isDest = dest;
        pn.physicalOut = bin == Bin::PhysicalOut;
        for (auto& f : fx) f->prepare(48000, kMaxBlock, 2);
        pn.fx = std::move(fx);
        pn.meterOffset = static_cast<int>(g->p.nodes.size()) * 4;
        g->p.nodes.push_back(std::move(pn));
        return n.get();
    };
    std::vector<NodeRuntime*> ins;
    for (int i = 0; i < inputs; ++i) {
        NodeRuntime* n = add(Bin::PhysicalIn, true, false, {createBuiltinFx("denoise"), createBuiltinFx("eq"), createBuiltinFx("compressor")});
        n->tone = std::make_shared<ToneGen>();
        n->tone->freq = 110.0f * static_cast<float>(i + 1);
        ins.push_back(n);
    }
    for (int i = 0; i < inputs; ++i) {
        add(Bin::PhysicalOut, false, true, {createBuiltinFx("eq"), createBuiltinFx("limiter")});
        auto r = std::make_shared<RouteRuntime>();
        g->routes.push_back(r);
        g->p.nodes.back().in.push_back({ins[static_cast<size_t>(i)], r.get(), {{0, 0, 1.0f}, {1, 1, 1.0f}}});
    }
    g->p.meters = std::make_shared<MeterBank>(g->p.nodes.size() * 4, 1);
    computeLevels(g->p);
    return g;
}

double run(Graph& g, DspScheduler* sched, int blocks) {
    for (int i = 0; i < 50; ++i) processProgram(g.p, 128, 0.0, sched);  // warm up
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < blocks; ++i) processProgram(g.p, 128, 0.0, sched);
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / blocks;
}
}  // namespace

int main(int argc, char** argv) {
    enableFlushToZero();
    const int inputs = argc > 1 ? std::atoi(argv[1]) : 8;
    const int blocks = argc > 2 ? std::atoi(argv[2]) : 2000;
    DspPool pool(DspPool::defaultWorkers());
    auto g = build(inputs);
    const double period = 128.0 / 48000.0 * 1e6;
    const double one = run(*g, nullptr, blocks);
    std::printf("graph: %d inputs (denoise+eq+compressor) -> %d outputs, 128-frame blocks (%.0f us)\n", inputs, inputs, period);
    std::printf("1 core : %8.1f us/block (%5.1f %% of the block)\n", one, one / period * 100);
    for (int threads = 2; threads <= pool.workers() + 1; threads *= 2) {
        DspScheduler s;
        s.pool = &pool;
        s.mode.store(threads);
        const double t = run(*g, &s, blocks);
        std::printf("%d cores: %8.1f us/block (%5.1f %% of the block), %.2fx\n", threads, t, t / period * 100, one / t);
    }
    DspScheduler aut;
    aut.pool = &pool;
    const double a = run(*g, &aut, blocks);
    std::printf("auto   : %8.1f us/block (%5.1f %% of the block), %.2fx, %d threads\n", a, a / period * 100, one / a, aut.threadsInUse.load());
    return 0;
}
