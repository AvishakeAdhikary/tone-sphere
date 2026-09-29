// ToneSphere Test Gain — a deterministic VST3 plugin for proving the host.
//
// MIT, like the rest of ToneSphere; built on the Steinberg VST3 SDK (MIT since 3.8).
//
// It does exactly three things, each of which a test can check from outside:
//   - multiplies by a gain: parameter 0, normalised 0..1 -> linear 0..2 (0.5 is unity);
//   - delays by exactly kLatency samples, and reports that latency, so "reported plugin
//     latency" can be compared with a measured one;
//   - saves and restores its gain through the component state stream.
//
// The module also exports "ToneSphere Test Crash", which dereferences null in initialize()
// — a stand-in for a plugin that crashes on load, for the host's failure-isolation tests.
// Nothing about either class is useful as audio processing; that is the point.
#include "base/source/fstreamer.h"
#include "pluginterfaces/base/ibstream.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "public.sdk/source/main/pluginfactory.h"
#include "public.sdk/source/vst/vstaudioeffect.h"
#include "public.sdk/source/vst/vsteditcontroller.h"

#include <algorithm>
#include <cstring>

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace {

constexpr ParamID kGain = 0;
constexpr int32 kLatency = 64;
constexpr int32 kMaxChannels = 8;
constexpr uint32 kStateMagic = 0x54534731;  // "TSG1"

const FUID kProcessorUID(0x5E1B7C01, 0x9A2F4D3E, 0x8B6C1D4F, 0x2A3B4C01);
const FUID kControllerUID(0x5E1B7C02, 0x9A2F4D3E, 0x8B6C1D4F, 0x2A3B4C02);
const FUID kCrashUID(0x5E1B7C03, 0x9A2F4D3E, 0x8B6C1D4F, 0x2A3B4C03);
const FUID kCrashInProcessUID(0x5E1B7C04, 0x9A2F4D3E, 0x8B6C1D4F, 0x2A3B4C04);

class Processor : public AudioEffect {
public:
    Processor() { setControllerClass(kControllerUID); }
    static FUnknown* create(void*) { return static_cast<IAudioProcessor*>(new Processor); }

    tresult PLUGIN_API initialize(FUnknown* context) SMTG_OVERRIDE {
        const tresult r = AudioEffect::initialize(context);
        if (r != kResultOk) return r;
        addAudioInput(STR16("Input"), SpeakerArr::kStereo);
        addAudioOutput(STR16("Output"), SpeakerArr::kStereo);
        return kResultOk;
    }

    tresult PLUGIN_API setBusArrangements(SpeakerArrangement* inputs, int32 numIns, SpeakerArrangement* outputs,
                                          int32 numOuts) SMTG_OVERRIDE {
        // Any width up to kMaxChannels, as long as input and output match.
        if (numIns != 1 || numOuts != 1 || inputs[0] != outputs[0]) return kResultFalse;
        const int32 channels = SpeakerArr::getChannelCount(inputs[0]);
        if (channels < 1 || channels > kMaxChannels) return kResultFalse;
        return AudioEffect::setBusArrangements(inputs, numIns, outputs, numOuts);
    }

    tresult PLUGIN_API canProcessSampleSize(int32 size) SMTG_OVERRIDE {
        return size == kSample32 ? kResultTrue : kResultFalse;
    }

    uint32 PLUGIN_API getLatencySamples() SMTG_OVERRIDE { return kLatency; }

    tresult PLUGIN_API setActive(TBool state) SMTG_OVERRIDE {
        std::memset(delay_, 0, sizeof(delay_));
        position_ = 0;
        return AudioEffect::setActive(state);
    }

    tresult PLUGIN_API process(ProcessData& data) SMTG_OVERRIDE {
        if (data.inputParameterChanges) {
            const int32 count = data.inputParameterChanges->getParameterCount();
            for (int32 i = 0; i < count; ++i) {
                IParamValueQueue* queue = data.inputParameterChanges->getParameterData(i);
                if (!queue || queue->getParameterId() != kGain) continue;
                ParamValue value;
                int32 offset;
                // The last point of the block wins: the test only needs block accuracy.
                if (queue->getPoint(queue->getPointCount() - 1, offset, value) == kResultTrue) gain_ = value;
            }
        }
        if (data.numInputs == 0 || data.numOutputs == 0 || data.numSamples == 0) return kResultOk;

        const int32 channels = std::min(data.inputs[0].numChannels, data.outputs[0].numChannels);
        const float gain = static_cast<float>(gain_ * 2.0);
        for (int32 i = 0; i < data.numSamples; ++i) {
            for (int32 c = 0; c < channels; ++c) {
                const float in = data.inputs[0].channelBuffers32[c][i];
                data.outputs[0].channelBuffers32[c][i] = delay_[c][position_] * gain;
                delay_[c][position_] = in;
            }
            position_ = (position_ + 1) % kLatency;
        }
        data.outputs[0].silenceFlags = 0;
        return kResultOk;
    }

    tresult PLUGIN_API setState(IBStream* state) SMTG_OVERRIDE {
        IBStreamer s(state, kLittleEndian);
        uint32 magic = 0;
        double gain = 0;
        if (!s.readInt32u(magic) || magic != kStateMagic || !s.readDouble(gain)) return kResultFalse;
        gain_ = std::clamp(gain, 0.0, 1.0);
        return kResultOk;
    }

    tresult PLUGIN_API getState(IBStream* state) SMTG_OVERRIDE {
        IBStreamer s(state, kLittleEndian);
        s.writeInt32u(kStateMagic);
        s.writeDouble(gain_);
        return kResultOk;
    }

private:
    double gain_ = 0.5;
    float delay_[kMaxChannels][kLatency] = {};
    int32 position_ = 0;
};

class Controller : public EditController {
public:
    static FUnknown* create(void*) { return static_cast<IEditController*>(new Controller); }

    tresult PLUGIN_API initialize(FUnknown* context) SMTG_OVERRIDE {
        const tresult r = EditController::initialize(context);
        if (r != kResultOk) return r;
        parameters.addParameter(STR16("Gain"), STR16("x"), 0, 0.5, ParameterInfo::kCanAutomate, kGain, 0, STR16("Gn"));
        return kResultOk;
    }

    tresult PLUGIN_API setComponentState(IBStream* state) SMTG_OVERRIDE {
        IBStreamer s(state, kLittleEndian);
        uint32 magic = 0;
        double gain = 0;
        if (!s.readInt32u(magic) || magic != kStateMagic || !s.readDouble(gain)) return kResultFalse;
        return setParamNormalized(kGain, gain);
    }

    ParamValue PLUGIN_API normalizedParamToPlain(ParamID id, ParamValue value) SMTG_OVERRIDE {
        return id == kGain ? value * 2.0 : EditController::normalizedParamToPlain(id, value);
    }
    ParamValue PLUGIN_API plainParamToNormalized(ParamID id, ParamValue plain) SMTG_OVERRIDE {
        return id == kGain ? plain / 2.0 : EditController::plainParamToNormalized(id, plain);
    }
};

class Crash : public AudioEffect {
public:
    static FUnknown* create(void*) { return static_cast<IAudioProcessor*>(new Crash); }
    tresult PLUGIN_API initialize(FUnknown*) SMTG_OVERRIDE {
        volatile int* nowhere = nullptr;
        *nowhere = 1;  // the fault a buggy plugin would have
        return kResultOk;
    }
};

// Loads and activates normally, then faults on its tenth process() call — on the audio
// thread, where a real plugin bug does the most damage.
class CrashInProcess : public Processor {
public:
    static FUnknown* create(void*) { return static_cast<IAudioProcessor*>(new CrashInProcess); }
    tresult PLUGIN_API process(ProcessData& data) SMTG_OVERRIDE {
        if (++calls_ == 10) {
            volatile int* nowhere = nullptr;
            *nowhere = 1;
        }
        return Processor::process(data);
    }

private:
    int calls_ = 0;
};

}  // namespace

BEGIN_FACTORY_DEF("Neural Nexus Studios", "https://github.com/AvishakeAdhikary/tone-sphere", "")

DEF_CLASS2(INLINE_UID_FROM_FUID(kProcessorUID), PClassInfo::kManyInstances, kVstAudioEffectClass,
           "ToneSphere Test Gain", Vst::kDistributable, "Fx|Tools", "1.0.0", kVstVersionString, Processor::create)

DEF_CLASS2(INLINE_UID_FROM_FUID(kControllerUID), PClassInfo::kManyInstances, kVstComponentControllerClass,
           "ToneSphere Test Gain Controller", 0, "", "1.0.0", kVstVersionString, Controller::create)

DEF_CLASS2(INLINE_UID_FROM_FUID(kCrashUID), PClassInfo::kManyInstances, kVstAudioEffectClass,
           "ToneSphere Test Crash", 0, "Fx|Tools", "1.0.0", kVstVersionString, Crash::create)

DEF_CLASS2(INLINE_UID_FROM_FUID(kCrashInProcessUID), PClassInfo::kManyInstances, kVstAudioEffectClass,
           "ToneSphere Test Crash In Process", 0, "Fx|Tools", "1.0.0", kVstVersionString, CrashInProcess::create)

END_FACTORY
