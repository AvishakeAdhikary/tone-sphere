#include "dsp/fx.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "dsp/biquad.h"

#ifdef TONESPHERE_WITH_RNNOISE
extern "C" {
#include "rnnoise.h"
}
#endif

namespace ts {

Fx::Fx(std::string type, std::string name, std::vector<ParamDef> defs)
    : type_(std::move(type)), name_(std::move(name)), defs_(std::move(defs)) {
    values_ = std::make_unique<std::atomic<float>[]>(defs_.size() ? defs_.size() : 1);
    for (size_t i = 0; i < defs_.size(); ++i) values_[i].store(defs_[i].def);
}

bool Fx::setParam(const std::string& pid, float v) {
    for (size_t i = 0; i < defs_.size(); ++i) {
        if (defs_[i].id == pid) {
            values_[i].store(std::clamp(v, defs_[i].min, defs_[i].max), std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

float Fx::paramById(const std::string& pid) const {
    for (size_t i = 0; i < defs_.size(); ++i)
        if (defs_[i].id == pid) return values_[i].load(std::memory_order_relaxed);
    return 0.0f;
}

void Fx::prepare(double sampleRate, int maxBlock, int channels) {
    sr_ = sampleRate;
    maxBlock_ = maxBlock;
    channels_ = channels;
}

namespace {

inline float dbToLin(float db) { return std::pow(10.0f, db / 20.0f); }
inline float linToDb(float v) { return v <= 1e-9f ? -180.0f : 20.0f * std::log10(v); }
inline float coef(double sr, float ms) { return ms <= 0 ? 0.0f : static_cast<float>(std::exp(-1.0 / (sr * ms * 0.001))); }

// Peak envelope across channels for one frame.
inline float framePeak(float* const* buf, int channels, int i) {
    float p = 0;
    for (int c = 0; c < channels; ++c) p = std::max(p, std::fabs(buf[c][i]));
    return p;
}

// ---------------------------------------------------------------- Noise gate
class GateFx final : public Fx {
public:
    GateFx()
        : Fx("gate", "Noise gate",
             {{"threshold", "Threshold", -80, 0, -45, "dB"},
              {"attack", "Attack", 0.1f, 50, 2, "ms", true},
              {"hold", "Hold", 0, 500, 80, "ms"},
              {"release", "Release", 5, 1000, 120, "ms", true},
              {"range", "Range", -80, 0, -60, "dB"}}) {}
    void process(float* const* buf, int ch, int n) override {
        const float thr = dbToLin(param(0));
        const float att = coef(sr_, param(1)), rel = coef(sr_, param(3));
        const int holdN = static_cast<int>(param(2) * 0.001 * sr_);
        const float floor = dbToLin(param(4));
        float minG = 1.0f;
        for (int i = 0; i < n; ++i) {
            const float p = framePeak(buf, ch, i);
            env_ = p > env_ ? p : env_ * 0.9995f;
            if (env_ >= thr) {
                hold_ = holdN;
                target_ = 1.0f;
            } else if (hold_ > 0) {
                --hold_;
            } else {
                target_ = floor;
            }
            const float k = target_ > gain_ ? att : rel;
            gain_ = target_ + (gain_ - target_) * k;
            for (int c = 0; c < ch; ++c) buf[c][i] *= gain_;
            minG = std::min(minG, gain_);
        }
        reductionDb.store(linToDb(minG), std::memory_order_relaxed);
    }

private:
    float env_ = 0, gain_ = 1, target_ = 1;
    int hold_ = 0;
};

// ---------------------------------------------------------------- Compressor
class CompressorFx final : public Fx {
public:
    CompressorFx()
        : Fx("compressor", "Compressor",
             {{"threshold", "Threshold", -60, 0, -18, "dB"},
              {"ratio", "Ratio", 1, 20, 3, ":1"},
              {"attack", "Attack", 0.1f, 100, 10, "ms", true},
              {"release", "Release", 10, 1000, 120, "ms", true},
              {"knee", "Knee", 0, 12, 6, "dB"},
              {"makeup", "Make-up", 0, 24, 0, "dB"}}) {}
    void process(float* const* buf, int ch, int n) override {
        const float thr = param(0), ratio = param(1), knee = param(4);
        const float att = coef(sr_, param(2)), rel = coef(sr_, param(3));
        const float makeup = dbToLin(param(5));
        float maxRed = 0;
        for (int i = 0; i < n; ++i) {
            const float lvl = linToDb(framePeak(buf, ch, i));
            float over = lvl - thr, red = 0;
            if (knee > 0 && std::fabs(over) <= knee / 2) {
                const float x = over + knee / 2;
                red = (1 / ratio - 1) * x * x / (2 * knee);
            } else if (over > 0) {
                red = (1 / ratio - 1) * over;
            }
            // red is <= 0 dB; smooth in dB domain
            const float k = red < envDb_ ? att : rel;
            envDb_ = red + (envDb_ - red) * k;
            const float g = dbToLin(envDb_) * makeup;
            for (int c = 0; c < ch; ++c) buf[c][i] *= g;
            maxRed = std::min(maxRed, envDb_);
        }
        reductionDb.store(maxRed, std::memory_order_relaxed);
    }

private:
    float envDb_ = 0;
};

// ---------------------------------------------------------------- Limiter (brickwall-ish, 1 ms lookahead)
class LimiterFx final : public Fx {
public:
    LimiterFx() : Fx("limiter", "Limiter", {{"ceiling", "Ceiling", -12, 0, -1, "dB"}, {"release", "Release", 10, 500, 60, "ms", true}}) {}
    void prepare(double sr, int maxBlock, int channels) override {
        Fx::prepare(sr, maxBlock, channels);
        la_ = std::max(1, static_cast<int>(sr * 0.001));
        delay_.assign(static_cast<size_t>(channels) * la_, 0.0f);
        pos_ = 0;
    }
    void process(float* const* buf, int ch, int n) override {
        const float ceil = dbToLin(param(0));
        const float rel = coef(sr_, param(1));
        const float att = coef(sr_, 0.3f);
        float minG = 1.0f;
        for (int i = 0; i < n; ++i) {
            const float p = framePeak(buf, ch, i);
            const float need = p > ceil ? ceil / p : 1.0f;
            gain_ = need < gain_ ? need + (gain_ - need) * att : need + (gain_ - need) * rel;
            gain_ = std::min(gain_, need);  // never exceed what this sample needs
            for (int c = 0; c < ch && c < channels_; ++c) {
                float& slot = delay_[static_cast<size_t>(c) * la_ + pos_];
                const float out = slot;
                slot = buf[c][i];
                buf[c][i] = std::clamp(out * gainHist(), -ceil, ceil);
            }
            pushGain(gain_);
            pos_ = (pos_ + 1) % la_;
            minG = std::min(minG, gain_);
        }
        reductionDb.store(linToDb(minG), std::memory_order_relaxed);
    }

private:
    // Applies the minimum gain seen over the lookahead window to the delayed sample.
    float gainHist() const { return histMin_; }
    void pushGain(float g) {
        hist_[histPos_] = g;
        histPos_ = (histPos_ + 1) % kHist;
        histMin_ = 1.0f;
        for (float h : hist_) histMin_ = std::min(histMin_, h);
    }
    static constexpr int kHist = 48;
    float hist_[kHist] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    int histPos_ = 0;
    float histMin_ = 1.0f;
    std::vector<float> delay_;
    int la_ = 48, pos_ = 0;
    float gain_ = 1.0f;
};

// ---------------------------------------------------------------- 4-band EQ + low cut
class EqFx final : public Fx {
public:
    EqFx()
        : Fx("eq", "Equalizer",
             {{"hpf", "Low cut", 20, 400, 20, "Hz", true},
              {"low", "Bass", -15, 15, 0, "dB"},
              {"mid", "Mids", -15, 15, 0, "dB"},
              {"midFreq", "Mid freq", 200, 5000, 1000, "Hz", true},
              {"presence", "Presence", -15, 15, 0, "dB"},
              {"high", "Treble", -15, 15, 0, "dB"}}) {}
    void process(float* const* buf, int ch, int n) override {
        float cur[6];
        for (int i = 0; i < 6; ++i) cur[i] = param(i);
        if (std::memcmp(cur, last_, sizeof(cur)) != 0) {
            std::memcpy(last_, cur, sizeof(cur));
            hp_.setHighPass(sr_, cur[0]);
            lo_.setLowShelf(sr_, 120, cur[1]);
            mid_.setPeaking(sr_, cur[3], 0.9, cur[2]);
            pres_.setPeaking(sr_, 4000, 1.0, cur[4]);
            hi_.setHighShelf(sr_, 9000, cur[5]);
        }
        const bool useHp = cur[0] > 21;
        for (int c = 0; c < ch && c < 8; ++c) {
            float* x = buf[c];
            for (int i = 0; i < n; ++i) {
                float v = x[i];
                if (useHp) v = hp_.tick(c, v);
                if (cur[1] != 0) v = lo_.tick(c, v);
                if (cur[2] != 0) v = mid_.tick(c, v);
                if (cur[4] != 0) v = pres_.tick(c, v);
                if (cur[5] != 0) v = hi_.tick(c, v);
                x[i] = v;
            }
        }
    }

private:
    Biquad hp_, lo_, mid_, pres_, hi_;
    float last_[6] = {-1, -1, -1, -1, -1, -1};
};

// ---------------------------------------------------------------- Ducker (sidechain)
class DuckerFx final : public Fx {
public:
    DuckerFx()
        : Fx("ducker", "Auto-duck",
             {{"amount", "Reduce by", 0, 40, 12, "dB"},
              {"threshold", "Trigger level", -60, 0, -35, "dB"},
              {"attack", "Attack", 1, 200, 20, "ms", true},
              {"release", "Release", 50, 2000, 400, "ms", true}}) {}
    void process(float* const* buf, int ch, int n) override {
        const float duck = dbToLin(-param(0));
        const float thr = dbToLin(param(1));
        const float att = coef(sr_, param(2)), rel = coef(sr_, param(3));
        float* const* sc = sidechain.load(std::memory_order_acquire);
        const int scCh = sidechainChannels.load(std::memory_order_relaxed);
        float minG = 1.0f;
        for (int i = 0; i < n; ++i) {
            float p = 0;
            if (sc)
                for (int c = 0; c < scCh; ++c) p = std::max(p, std::fabs(sc[c][i]));
            env_ = p > env_ ? p : env_ * 0.9997f;
            const float target = env_ >= thr ? duck : 1.0f;
            const float k = target < gain_ ? att : rel;
            gain_ = target + (gain_ - target) * k;
            for (int c = 0; c < ch; ++c) buf[c][i] *= gain_;
            minG = std::min(minG, gain_);
        }
        reductionDb.store(linToDb(minG), std::memory_order_relaxed);
    }

private:
    float env_ = 0, gain_ = 1;
};

// ---------------------------------------------------------------- Delay (sync)
class DelayFx final : public Fx {
public:
    DelayFx() : Fx("delay", "Delay", {{"time", "Time", 0, 2000, 0, "ms"}}) {}
    void prepare(double sr, int maxBlock, int channels) override {
        Fx::prepare(sr, maxBlock, channels);
        cap_ = static_cast<int>(sr * 2.1) + maxBlock;
        line_.assign(static_cast<size_t>(channels) * cap_, 0.0f);
        w_ = 0;
    }
    void process(float* const* buf, int ch, int n) override {
        const int d = std::clamp(static_cast<int>(param(0) * 0.001 * sr_), 0, cap_ - 1);
        if (d == 0) return;
        for (int i = 0; i < n; ++i) {
            const int r = (w_ - d + cap_) % cap_;
            for (int c = 0; c < ch && c < channels_; ++c) {
                float* line = &line_[static_cast<size_t>(c) * cap_];
                line[w_] = buf[c][i];
                buf[c][i] = line[r];
            }
            w_ = (w_ + 1) % cap_;
        }
    }

private:
    std::vector<float> line_;
    int cap_ = 0, w_ = 0;
};

// ---------------------------------------------------------------- Noise suppression (RNNoise)
class DenoiseFx final : public Fx {
public:
    DenoiseFx()
        : Fx("denoise", "Noise suppression",
             {{"amount", "Amount", 0, 100, 100, "%"}, {"vadGate", "Voice gate", 0, 95, 0, "%"}}) {}
    ~DenoiseFx() override {
#ifdef TONESPHERE_WITH_RNNOISE
        for (auto* s : st_) rnnoise_destroy(s);
#endif
    }
    void prepare(double sr, int maxBlock, int channels) override {
        Fx::prepare(sr, maxBlock, channels);
#ifdef TONESPHERE_WITH_RNNOISE
        for (auto* s : st_) rnnoise_destroy(s);
        st_.clear();
        for (int c = 0; c < channels; ++c) st_.push_back(rnnoise_create(nullptr));
#endif
        in_.assign(static_cast<size_t>(channels) * kFrame, 0.0f);
        out_.assign(static_cast<size_t>(channels) * kFrame * 2, 0.0f);
        fill_ = 0;
    }
    void process(float* const* buf, int ch, int n) override {
#ifdef TONESPHERE_WITH_RNNOISE
        if (std::fabs(sr_ - 48000.0) > 1.0 || ch > static_cast<int>(st_.size())) return;  // RNNoise is 48 kHz only
        const float wet = param(0) / 100.0f;
        const float vadThr = param(1) / 100.0f;
        // Fixed 480-sample frames: buffer input, emit processed output with one frame of latency.
        for (int i = 0; i < n; ++i) {
            for (int c = 0; c < ch; ++c) {
                const float x = buf[c][i];
                in_[static_cast<size_t>(c) * kFrame + fill_] = x;
                buf[c][i] = out_[static_cast<size_t>(c) * kFrame + fill_];
            }
            if (++fill_ == kFrame) {
                fill_ = 0;
                float voice = 0;
                for (int c = 0; c < ch; ++c) {
                    float frame[kFrame], dry[kFrame];
                    for (int k = 0; k < kFrame; ++k) {
                        dry[k] = in_[static_cast<size_t>(c) * kFrame + k];
                        frame[k] = dry[k] * 32768.0f;
                    }
                    const float vad = rnnoise_process_frame(st_[static_cast<size_t>(c)], frame, frame);
                    voice = std::max(voice, vad);
                    const float g = vadThr > 0 && vad < vadThr ? 0.0f : 1.0f;
                    for (int k = 0; k < kFrame; ++k)
                        out_[static_cast<size_t>(c) * kFrame + k] = g * (wet * frame[k] / 32768.0f + (1 - wet) * dry[k]);
                }
                reductionDb.store(voice < 0.5f ? -20.0f : 0.0f, std::memory_order_relaxed);
            }
        }
#else
        (void)buf;
        (void)ch;
        (void)n;
#endif
    }

private:
    static constexpr int kFrame = 480;
#ifdef TONESPHERE_WITH_RNNOISE
    std::vector<DenoiseState*> st_;
#endif
    std::vector<float> in_, out_;
    int fill_ = 0;
};

}  // namespace

std::shared_ptr<Fx> createBuiltinFx(const std::string& type) {
    if (type == "gate") return std::make_shared<GateFx>();
    if (type == "compressor") return std::make_shared<CompressorFx>();
    if (type == "limiter") return std::make_shared<LimiterFx>();
    if (type == "eq") return std::make_shared<EqFx>();
    if (type == "ducker") return std::make_shared<DuckerFx>();
    if (type == "delay") return std::make_shared<DelayFx>();
    if (type == "denoise") return std::make_shared<DenoiseFx>();
    return nullptr;
}

}  // namespace ts
