#include "backends/win/peak_meters.h"

#include <windows.h>

#include <audiopolicy.h>
#include <endpointvolume.h>
#include <mmdeviceapi.h>
#include <wrl/client.h>

#include <algorithm>
#include <map>
#include <mutex>

#include "core/graph.h"
#include "util/platform.h"

using Microsoft::WRL::ComPtr;

namespace ts {

struct PeakMeters::Impl {
    std::mutex mu;
    std::vector<Target> pending;
    bool dirty = false;
    bool com = false;
    struct Live {
        int offset, channels;
        std::vector<ComPtr<IAudioMeterInformation>> meters;
    };
    std::vector<Live> live;

    void rebuild(const std::vector<Target>& targets) {
        live.clear();
        ComPtr<IMMDeviceEnumerator> en;
        if (FAILED(CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, IID_PPV_ARGS(&en)))) return;
        // pid -> session meters (render and capture), collected once per rebuild.
        std::map<uint32_t, std::vector<ComPtr<IAudioMeterInformation>>> sessions[2];
        bool needSessions = std::any_of(targets.begin(), targets.end(), [](const Target& t) { return !t.pids.empty(); });
        if (needSessions) {
            for (int flow = 0; flow < 2; ++flow) {
                ComPtr<IMMDeviceCollection> coll;
                if (FAILED(en->EnumAudioEndpoints(flow ? eCapture : eRender, DEVICE_STATE_ACTIVE, &coll))) continue;
                UINT n = 0;
                coll->GetCount(&n);
                for (UINT i = 0; i < n; ++i) {
                    ComPtr<IMMDevice> dev;
                    ComPtr<IAudioSessionManager2> mgr;
                    ComPtr<IAudioSessionEnumerator> se;
                    if (FAILED(coll->Item(i, &dev)) || FAILED(dev->Activate(__uuidof(IAudioSessionManager2), CLSCTX_ALL, nullptr, &mgr)) ||
                        FAILED(mgr->GetSessionEnumerator(&se)))
                        continue;
                    int count = 0;
                    se->GetCount(&count);
                    for (int k = 0; k < count; ++k) {
                        ComPtr<IAudioSessionControl> ctl;
                        ComPtr<IAudioSessionControl2> ctl2;
                        ComPtr<IAudioMeterInformation> meter;
                        if (FAILED(se->GetSession(k, &ctl)) || FAILED(ctl.As(&ctl2)) || FAILED(ctl.As(&meter))) continue;
                        DWORD pid = 0;
                        ctl2->GetProcessId(&pid);
                        if (pid) sessions[flow][pid].push_back(meter);
                    }
                }
            }
        }
        for (const Target& t : targets) {
            Live l{t.offset, t.channels, {}};
            if (!t.endpointId.empty()) {
                ComPtr<IMMDevice> dev;
                ComPtr<IAudioMeterInformation> m;
                const std::wstring id = toWide(t.endpointId);
                if (SUCCEEDED(en->GetDevice(id.c_str(), &dev)) &&
                    SUCCEEDED(dev->Activate(__uuidof(IAudioMeterInformation), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(m.GetAddressOf()))))
                    l.meters.push_back(m);
            }
            for (uint32_t pid : t.pids) {
                auto& map = sessions[t.capture ? 1 : 0];
                if (auto it = map.find(pid); it != map.end()) l.meters.insert(l.meters.end(), it->second.begin(), it->second.end());
            }
            if (!l.meters.empty()) live.push_back(std::move(l));
        }
    }
};

PeakMeters::PeakMeters() : impl_(std::make_unique<Impl>()) {}
PeakMeters::~PeakMeters() = default;

void PeakMeters::setTargets(std::vector<Target> targets) {
    std::lock_guard lk(impl_->mu);
    impl_->pending = std::move(targets);
    impl_->dirty = true;
}

void PeakMeters::poll(MeterBank& bank, uint32_t) {
    Impl& s = *impl_;
    if (!s.com) {
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        s.com = true;
    }
    {
        std::vector<Target> t;
        bool dirty = false;
        {
            std::lock_guard lk(s.mu);
            if (s.dirty) {
                t = s.pending;
                s.dirty = false;
                dirty = true;
            }
        }
        if (dirty) s.rebuild(t);
    }
    float peaks[kMaxChannels];
    for (auto& l : s.live) {
        const int ch = std::min(l.channels, kMaxChannels);
        std::fill(peaks, peaks + ch, 0.0f);
        for (auto& m : l.meters) {
            UINT mc = 0;
            float tmp[kMaxChannels] = {};
            if (SUCCEEDED(m->GetMeteringChannelCount(&mc)) && mc > 0 &&
                SUCCEEDED(m->GetChannelsPeakValues(std::min<UINT>(mc, kMaxChannels), tmp))) {
                for (int c = 0; c < ch; ++c) peaks[c] = std::max(peaks[c], tmp[std::min<UINT>(static_cast<UINT>(c), mc - 1)]);
            } else {
                float p = 0;
                if (SUCCEEDED(m->GetPeakValue(&p)))
                    for (int c = 0; c < ch; ++c) peaks[c] = std::max(peaks[c], p);
            }
        }
        for (int c = 0; c < ch; ++c) {
            const size_t i = static_cast<size_t>(l.offset + c * 2);
            if (i + 1 >= bank.size) break;
            bank.values[i].store(peaks[c], std::memory_order_relaxed);
            bank.values[i + 1].store(peaks[c] * 0.7071f, std::memory_order_relaxed);
        }
    }
}

}  // namespace ts
