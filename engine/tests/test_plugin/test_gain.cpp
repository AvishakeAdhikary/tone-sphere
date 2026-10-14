// "ToneSphere Test Gain": a tiny VST3 effect used by the automated tests.
// One parameter (Gain, -24..+12 dB, default -6 dB), mono or stereo, state save/restore,
// and a native editor window with a slider (class "ToneSphereTestGainView") that sends
// edits through the host like any real plugin editor.
#include <cmath>
#include <cstring>

#include <windows.h>
#include <commctrl.h>

#include "base/source/fstreamer.h"
#include "pluginterfaces/base/ibstream.h"
#include "pluginterfaces/gui/iplugview.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "public.sdk/source/common/pluginview.h"
#include "public.sdk/source/main/pluginfactory.h"
#include "public.sdk/source/vst/vstsinglecomponenteffect.h"

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace {

const FUID kTestGainUID(0x4E78546F, 0x6E655465, 0x73744761, 0x696E3031);
constexpr ParamID kGainId = 0;
constexpr int kSliderId = 100;

float gainLinear(double norm) { return std::pow(10.0f, (-24.0f + 36.0f * static_cast<float>(norm)) / 20.0f); }

class TestGain;

class GainView : public CPluginView {
public:
    explicit GainView(TestGain* fx) : CPluginView(nullptr), fx_(fx) { rect = ViewRect(0, 0, 360, 90); }
    tresult PLUGIN_API isPlatformTypeSupported(FIDString type) override {
        return std::strcmp(type, kPlatformTypeHWND) == 0 ? kResultTrue : kResultFalse;
    }
    tresult PLUGIN_API attached(void* parent, FIDString type) override;
    tresult PLUGIN_API removed() override {
        if (container_) DestroyWindow(container_);
        container_ = slider_ = nullptr;
        return CPluginView::removed();
    }
    void setValue(double v) {
        if (slider_) SendMessageW(slider_, TBM_SETPOS, TRUE, static_cast<LPARAM>(v * 1000));
    }

private:
    static LRESULT CALLBACK proc(HWND h, UINT msg, WPARAM wp, LPARAM lp);
    TestGain* fx_;
    HWND container_ = nullptr, slider_ = nullptr;
};

class TestGain : public SingleComponentEffect {
public:
    static FUnknown* create(void*) { return static_cast<IAudioProcessor*>(new TestGain()); }

    tresult PLUGIN_API initialize(FUnknown* ctx) override {
        const tresult r = SingleComponentEffect::initialize(ctx);
        if (r != kResultOk) return r;
        addAudioInput(STR16("Input"), SpeakerArr::kStereo);
        addAudioOutput(STR16("Output"), SpeakerArr::kStereo);
        parameters.addParameter(STR16("Gain"), STR16("dB"), 0, 0.5, ParameterInfo::kCanAutomate, kGainId);
        return kResultOk;
    }

    tresult PLUGIN_API setBusArrangements(SpeakerArrangement* ins, int32 nIns, SpeakerArrangement* outs, int32 nOuts) override {
        if (nIns == 1 && nOuts == 1 && ins[0] == outs[0] && (ins[0] == SpeakerArr::kMono || ins[0] == SpeakerArr::kStereo)) {
            return SingleComponentEffect::setBusArrangements(ins, nIns, outs, nOuts);
        }
        return kResultFalse;
    }

    tresult PLUGIN_API canProcessSampleSize(int32 size) override { return size == kSample32 ? kResultTrue : kResultFalse; }

    tresult PLUGIN_API process(ProcessData& data) override {
        if (IParameterChanges* pc = data.inputParameterChanges) {
            for (int32 i = 0; i < pc->getParameterCount(); ++i) {
                IParamValueQueue* q = pc->getParameterData(i);
                int32 off;
                ParamValue v;
                if (q && q->getParameterId() == kGainId && q->getPoint(q->getPointCount() - 1, off, v) == kResultOk) gain_ = v;
            }
        }
        if (data.numInputs == 0 || data.numOutputs == 0 || data.numSamples <= 0) return kResultOk;
        const float g = gainLinear(gain_);
        const int32 ch = std::min(data.inputs[0].numChannels, data.outputs[0].numChannels);
        for (int32 c = 0; c < ch; ++c) {
            const float* in = data.inputs[0].channelBuffers32[c];
            float* out = data.outputs[0].channelBuffers32[c];
            for (int32 i = 0; i < data.numSamples; ++i) out[i] = in[i] * g;
        }
        return kResultOk;
    }

    tresult PLUGIN_API setState(IBStream* s) override {
        IBStreamer st(s, kLittleEndian);
        double v;
        if (!st.readDouble(v)) return kResultFalse;
        gain_ = v;
        setParamNormalized(kGainId, v);
        if (view_) view_->setValue(v);
        return kResultOk;
    }
    tresult PLUGIN_API getState(IBStream* s) override {
        IBStreamer st(s, kLittleEndian);
        return st.writeDouble(getParamNormalized(kGainId)) ? kResultOk : kResultFalse;
    }

    IPlugView* PLUGIN_API createView(FIDString name) override {
        if (std::strcmp(name, ViewType::kEditor) != 0) return nullptr;
        view_ = new GainView(this);
        return view_;
    }

    void editFromView(double v) {
        beginEdit(kGainId);
        setParamNormalized(kGainId, v);
        performEdit(kGainId, v);
        endEdit(kGainId);
    }
    void viewGone() { view_ = nullptr; }
    double current() { return getParamNormalized(kGainId); }

private:
    double gain_ = 0.5;
    GainView* view_ = nullptr;
};

tresult PLUGIN_API GainView::attached(void* parent, FIDString type) {
    static bool reg = [] {
        INITCOMMONCONTROLSEX icc{sizeof(icc), ICC_BAR_CLASSES};
        InitCommonControlsEx(&icc);
        WNDCLASSW wc{};
        wc.lpfnWndProc = proc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        wc.lpszClassName = L"ToneSphereTestGainView";
        return RegisterClassW(&wc) != 0;
    }();
    (void)reg;
    container_ = CreateWindowExW(0, L"ToneSphereTestGainView", L"", WS_CHILD | WS_VISIBLE, 0, 0, rect.getWidth(), rect.getHeight(),
                                 static_cast<HWND>(parent), nullptr, GetModuleHandleW(nullptr), this);
    CreateWindowExW(0, L"STATIC", L"ToneSphere Test Gain", WS_CHILD | WS_VISIBLE, 12, 10, 300, 20, container_, nullptr, nullptr, nullptr);
    slider_ = CreateWindowExW(0, TRACKBAR_CLASSW, L"", WS_CHILD | WS_VISIBLE | TBS_HORZ, 12, 40, 330, 30, container_,
                              reinterpret_cast<HMENU>(static_cast<INT_PTR>(kSliderId)), nullptr, nullptr);
    SendMessageW(slider_, TBM_SETRANGE, TRUE, MAKELPARAM(0, 1000));
    setValue(fx_->current());
    return CPluginView::attached(parent, type);
}

LRESULT CALLBACK GainView::proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_NCCREATE) SetWindowLongPtrW(h, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams));
    auto* self = reinterpret_cast<GainView*>(GetWindowLongPtrW(h, GWLP_USERDATA));
    if (msg == WM_HSCROLL && self && self->slider_) {
        const double v = static_cast<double>(SendMessageW(self->slider_, TBM_GETPOS, 0, 0)) / 1000.0;
        self->fx_->editFromView(v);
        return 0;
    }
    if (msg == WM_DESTROY && self) self->fx_->viewGone();
    return DefWindowProcW(h, msg, wp, lp);
}

}  // namespace

BEGIN_FACTORY_DEF("ToneSphere", "https://github.com/AvishakeAdhikary/tone-sphere", "mailto:tonesphere@example.invalid")
DEF_CLASS2(INLINE_UID_FROM_FUID(kTestGainUID), PClassInfo::kManyInstances, kVstAudioEffectClass, "ToneSphere Test Gain", 0, "Fx",
           "1.0.0", kVstVersionString, TestGain::create)
END_FACTORY
