#include "control/remote.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "core/engine.h"
#include "util/log.h"
#include "util/platform.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winmm.lib")
using SocketT = SOCKET;
static constexpr SocketT kNoSocket = INVALID_SOCKET;
static void closeSocket(SocketT s) { closesocket(s); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
using SocketT = int;
static constexpr SocketT kNoSocket = -1;
static void closeSocket(SocketT s) {
    ::shutdown(s, SHUT_RDWR);
    ::close(s);
}
#endif

using nlohmann::json;

namespace ts {

// ================================================================ pure helpers (unit tested)
namespace {

size_t pad4(size_t n) { return (n + 3) & ~size_t(3); }

bool readString(const uint8_t* d, size_t len, size_t& pos, std::string& out) {
    const void* z = std::memchr(d + pos, 0, len - pos);
    if (!z) return false;
    const size_t n = static_cast<const uint8_t*>(z) - (d + pos);
    out.assign(reinterpret_cast<const char*>(d + pos), n);
    pos += pad4(n + 1);
    return pos <= len;
}

uint32_t be32(const uint8_t* p) { return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]; }

bool parseMessage(const uint8_t* d, size_t len, std::vector<std::pair<std::string, std::vector<OscArg>>>& out) {
    size_t pos = 0;
    std::string address, tags;
    if (!readString(d, len, pos, address) || address.empty() || address[0] != '/') return false;
    std::vector<OscArg> args;
    if (pos < len) {
        if (!readString(d, len, pos, tags) || tags.empty() || tags[0] != ',') return false;
        for (size_t t = 1; t < tags.size(); ++t) {
            OscArg a;
            a.type = tags[t];
            switch (a.type) {
                case 'i':
                    if (pos + 4 > len) return false;
                    a.num = static_cast<int32_t>(be32(d + pos));
                    pos += 4;
                    break;
                case 'f': {
                    if (pos + 4 > len) return false;
                    const uint32_t u = be32(d + pos);
                    float f;
                    std::memcpy(&f, &u, 4);
                    a.num = f;
                    pos += 4;
                    break;
                }
                case 'd': {
                    if (pos + 8 > len) return false;
                    const uint64_t u = (uint64_t(be32(d + pos)) << 32) | be32(d + pos + 4);
                    double v;
                    std::memcpy(&v, &u, 8);
                    a.num = v;
                    pos += 8;
                    break;
                }
                case 's':
                    if (!readString(d, len, pos, a.str)) return false;
                    break;
                case 'T':
                    a.num = 1;
                    break;
                case 'F':
                    a.num = 0;
                    break;
                default:
                    return false;  // unsupported type: ignore the whole message
            }
            args.push_back(std::move(a));
        }
    }
    out.emplace_back(std::move(address), std::move(args));
    return true;
}

std::optional<float> numberArg(const std::vector<OscArg>& args, size_t i) {
    if (i < args.size() && args[i].type != 's') return static_cast<float>(args[i].num);
    return std::nullopt;
}

}  // namespace

bool parseOscPacket(const uint8_t* d, size_t len, std::vector<std::pair<std::string, std::vector<OscArg>>>& out) {
    if (len >= 16 && std::memcmp(d, "#bundle", 8) == 0) {
        size_t pos = 16;  // "#bundle\0" + 8-byte time tag
        bool any = false;
        while (pos + 4 <= len) {
            const uint32_t n = be32(d + pos);
            pos += 4;
            if (n > len - pos) return any;
            any |= parseOscPacket(d + pos, n, out);
            pos += n;
        }
        return any;
    }
    return parseMessage(d, len, out);
}

std::optional<RemoteAction> oscToAction(const std::string& address, const std::vector<OscArg>& args) {
    // "/nextone/" is the prefix from before the rename: controllers set up then keep working.
    std::string verb;
    for (const char* prefix : {"/tonesphere/", "/nextone/"}) {
        const size_t n = std::strlen(prefix);
        if (address.compare(0, n, prefix) == 0) verb = address.substr(n);
    }
    if (verb.empty()) return std::nullopt;
    RemoteAction a;
    a.action = verb;
    if (verb == "volume" || verb == "gain") {
        if (args.empty() || args[0].type != 's' || !numberArg(args, 1)) return std::nullopt;
        a.target = args[0].str;
        a.value = numberArg(args, 1);
    } else if (verb == "mute" || verb == "solo") {
        if (args.empty() || args[0].type != 's') return std::nullopt;
        a.target = args[0].str;
        a.value = numberArg(args, 1);
    } else if (verb == "scene") {
        if (args.empty() || args[0].type != 's') return std::nullopt;
        a.target = args[0].str;
    } else if (verb == "record") {
        a.value = numberArg(args, 0);
    } else if (verb != "replay") {
        return std::nullopt;
    }
    return a;
}

float faderToDb(float pos) {
    pos = std::clamp(pos, 0.0f, 1.0f);
    return pos <= 0.0f ? -90.0f : -60.0f + 66.0f * pos;
}

std::optional<RemoteAction> midiToAction(const json& mappings, uint8_t status, uint8_t d1, uint8_t d2) {
    const int kind = status & 0xF0, channel = (status & 0x0F) + 1;
    const bool isCc = kind == 0xB0, isNoteOn = kind == 0x90 && d2 > 0;
    if (!isCc && !isNoteOn) return std::nullopt;
    if (!mappings.is_array()) return std::nullopt;
    for (const json& m : mappings) {
        if (m.value("type", "cc") != (isCc ? "cc" : "note")) continue;
        if (m.value("number", -1) != d1) continue;
        const int ch = m.value("channel", 0);
        if (ch != 0 && ch != channel) continue;
        RemoteAction a;
        a.action = m.value("action", "");
        a.target = m.value("target", "");
        const float v = d2 / 127.0f;
        if (a.action == "volume") {
            a.value = v;
        } else if (a.action == "gain") {
            a.value = faderToDb(v);
            a.action = "gainDb";
        } else if (a.action == "mute" || a.action == "solo" || a.action == "record") {
            if (isCc) a.value = v >= 0.5f ? 1.0f : 0.0f;  // knob/switch position; a note toggles
        } else if (a.action == "scene" || a.action == "replay") {
            if (isCc && v < 0.5f) return std::nullopt;  // trigger on press only
        } else {
            continue;
        }
        return a;
    }
    return std::nullopt;
}

// ================================================================ runtime
struct RemoteControl::Impl {
    Engine& engine;
    std::thread worker, oscThread;
    std::atomic<bool> running{true};
    std::mutex mu;
    std::condition_variable cv;
    std::deque<RemoteAction> actions;
    json mappings = json::array();
    int oscPort = 0;
    std::atomic<SocketT> sock{kNoSocket};

    // MIDI learn
    std::mutex learnMu;
    std::condition_variable learnCv;
    bool learning = false;
    std::optional<json> learned;

#ifdef _WIN32
    std::vector<HMIDIIN> midiIns;
    bool midiWanted = false;
#endif

    explicit Impl(Engine& e) : engine(e) {
#ifdef _WIN32
        WSADATA w;
        WSAStartup(MAKEWORD(2, 2), &w);
#endif
        worker = std::thread([this] { work(); });
    }

    ~Impl() {
        running = false;
        cv.notify_all();
        learnCv.notify_all();
        closeOsc();
        if (oscThread.joinable()) oscThread.join();
        if (worker.joinable()) worker.join();
#ifdef _WIN32
        closeMidi();
        WSACleanup();
#endif
    }

    void push(RemoteAction a) {
        std::lock_guard lk(mu);
        actions.push_back(std::move(a));
        cv.notify_one();
    }

    // ---- OSC
    void openOsc(int port) {
        closeOsc();
        if (oscThread.joinable()) oscThread.join();
        if (port <= 0) return;
        SocketT s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (s == kNoSocket) return;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(static_cast<uint16_t>(port));
        a.sin_addr.s_addr = htonl(INADDR_ANY);  // phones/tablets on the network (opt-in setting)
        if (::bind(s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0) {
            TS_WARN("OSC: port %d is in use", port);
            closeSocket(s);
            return;
        }
        sock = s;
        TS_INFO("OSC control listening on UDP port %d", port);
        oscThread = std::thread([this, s] {
            setThreadName("ts-osc");
            std::vector<uint8_t> buf(65536);
            while (running && sock.load() == s) {
                const auto n = ::recv(s, reinterpret_cast<char*>(buf.data()), static_cast<int>(buf.size()), 0);
                if (n <= 0) break;  // socket closed
                std::vector<std::pair<std::string, std::vector<OscArg>>> msgs;
                if (!parseOscPacket(buf.data(), static_cast<size_t>(n), msgs)) continue;
                for (auto& [addr, args] : msgs)
                    if (auto a = oscToAction(addr, args)) push(*a);
            }
        });
    }

    void closeOsc() {
        const SocketT s = sock.exchange(kNoSocket);
        if (s != kNoSocket) closeSocket(s);
    }

    // ---- MIDI (Windows)
#ifdef _WIN32
    static void CALLBACK midiProc(HMIDIIN, UINT msg, DWORD_PTR inst, DWORD_PTR p1, DWORD_PTR) {
        if (msg != MIM_DATA) return;
        auto* self = reinterpret_cast<Impl*>(inst);
        const uint8_t status = p1 & 0xFF, d1 = (p1 >> 8) & 0x7F, d2 = (p1 >> 16) & 0x7F;
        self->onMidi(status, d1, d2);
    }

    void openMidi() {
        if (!midiIns.empty()) return;
        const UINT n = midiInGetNumDevs();
        for (UINT i = 0; i < n; ++i) {
            HMIDIIN h = nullptr;
            if (midiInOpen(&h, i, reinterpret_cast<DWORD_PTR>(&midiProc), reinterpret_cast<DWORD_PTR>(this), CALLBACK_FUNCTION) == MMSYSERR_NOERROR) {
                midiInStart(h);
                midiIns.push_back(h);
            }
        }
        if (!midiIns.empty()) TS_INFO("MIDI control: %zu input device(s)", midiIns.size());
    }

    void closeMidi() {
        for (HMIDIIN h : midiIns) {
            midiInStop(h);
            midiInReset(h);
            midiInClose(h);
        }
        midiIns.clear();
    }
#endif

    void onMidi(uint8_t status, uint8_t d1, uint8_t d2) {
        const int kind = status & 0xF0;
        {
            std::lock_guard lk(learnMu);
            if (learning && (kind == 0xB0 || (kind == 0x90 && d2 > 0))) {
                learned = json{{"type", kind == 0xB0 ? "cc" : "note"}, {"channel", (status & 0x0F) + 1}, {"number", d1}, {"value", d2}};
                learning = false;
                learnCv.notify_all();
                return;
            }
        }
        json maps;
        {
            std::lock_guard lk(mu);
            maps = mappings;
        }
        if (auto a = midiToAction(maps, status, d1, d2)) push(*a);
    }

    // ---- settings + actions
    void reload() {
        json st;
        try {
            st = engine.state();
        } catch (...) {
            return;
        }
        const json& s = st["settings"];
        const int port = s.value("oscPort", 0);
        json maps = s.contains("midiMappings") ? s["midiMappings"] : json::array();
        bool portChanged;
        {
            std::lock_guard lk(mu);
            portChanged = port != oscPort;
            oscPort = port;
            mappings = maps;
        }
        if (portChanged) openOsc(port);
#ifdef _WIN32
        bool learningNow;
        {
            std::lock_guard lk(learnMu);
            learningNow = learning;
        }
        const bool want = (maps.is_array() && !maps.empty()) || learningNow;
        if (want) openMidi();
        else if (!want && !midiIns.empty()) closeMidi();
#endif
    }

    std::string resolveNode(const json& st, const std::string& t) {
        auto lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return s;
        };
        const std::string lt = lower(t);
        std::string partial;
        int partials = 0;
        for (const json& n : st["nodes"]) {
            const std::string id = n.value("id", ""), name = lower(n.value("name", ""));
            if (id == t || name == lt) return id;
            if (!lt.empty() && name.find(lt) != std::string::npos) {
                partial = id;
                ++partials;
            }
        }
        if (partials == 1) return partial;
        throw std::runtime_error("no single device or mix named \"" + t + "\"");
    }

    void run(const RemoteAction& a) {
        try {
            const json st = engine.state();
            if (a.action == "volume" || a.action == "gain" || a.action == "gainDb") {
                const float db = a.action == "volume" ? faderToDb(*a.value) : std::clamp(*a.value, -90.0f, 12.0f);
                engine.call("node.update", {{"id", resolveNode(st, a.target)}, {"gainDb", db}});
            } else if (a.action == "mute" || a.action == "solo") {
                const std::string id = resolveNode(st, a.target);
                const char* key = a.action == "mute" ? "muted" : "solo";
                bool next = true;
                if (a.value) next = *a.value >= 0.5f;
                else
                    for (const json& n : st["nodes"])
                        if (n["id"] == id) next = !n.value(key, false);
                engine.call("node.update", {{"id", id}, {key, next}});
            } else if (a.action == "scene") {
                std::string sid;
                for (const json& sc : st["scenes"])
                    if (sc["id"] == a.target || sc.value("name", "") == a.target) sid = sc["id"];
                if (sid.empty()) throw std::runtime_error("no scene \"" + a.target + "\"");
                engine.call("scene.load", {{"id", sid}});
            } else if (a.action == "record") {
                const bool recording = st["recorder"]["recording"].get<bool>();
                const bool want = a.value ? *a.value >= 0.5f : !recording;
                if (want && !recording) {
                    json ids = json::array();
                    for (const json& n : st["nodes"])
                        if (n["bin"] == "bus") ids.push_back(n["id"]);
                    if (ids.empty())
                        for (const json& n : st["nodes"])
                            if (n["bin"] == "physical-out" && n["state"] != "unavailable") ids.push_back(n["id"]);
                    engine.call("recorder.start", {{"nodeIds", ids}, {"format", "wav"}});
                } else if (!want && recording) {
                    engine.call("recorder.stop", json::object());
                }
            } else if (a.action == "replay") {
                engine.call("recorder.saveReplay", json::object());
            }
        } catch (const std::exception& e) {
            TS_WARN("Remote control (%s %s): %s", a.action.c_str(), a.target.c_str(), e.what());
        }
    }

    void work() {
        setThreadName("ts-remote");
        double lastReload = -10;
        while (running) {
            std::optional<RemoteAction> a;
            {
                std::unique_lock lk(mu);
                cv.wait_for(lk, std::chrono::milliseconds(250), [this] { return !actions.empty() || !running; });
                if (!running) break;
                if (!actions.empty()) {
                    a = actions.front();
                    actions.pop_front();
                }
            }
            if (a) run(*a);
            if (nowSeconds() - lastReload > 2.0) {
                lastReload = nowSeconds();
                reload();
            }
        }
    }
};

RemoteControl::RemoteControl(Engine& engine) : impl_(std::make_unique<Impl>(engine)) {}
RemoteControl::~RemoteControl() = default;

json RemoteControl::learn(double seconds) {
    {
        std::lock_guard lk(impl_->learnMu);
        impl_->learning = true;
        impl_->learned.reset();
    }
    // The worker opens the MIDI inputs at its next settings reload (within 2 s).
    std::unique_lock lk(impl_->learnMu);
    const bool got = impl_->learnCv.wait_for(lk, std::chrono::milliseconds(static_cast<int>(seconds * 1000)), [this] { return impl_->learned.has_value(); });
    impl_->learning = false;
    if (!got) throw std::runtime_error("No MIDI control moved. Move a knob, fader or pad on your controller and try again.");
    return *impl_->learned;
}

json RemoteControl::midiDevices() {
    json out = json::array();
#ifdef _WIN32
    const UINT n = midiInGetNumDevs();
    for (UINT i = 0; i < n; ++i) {
        MIDIINCAPSW caps{};
        if (midiInGetDevCapsW(i, &caps, sizeof(caps)) == MMSYSERR_NOERROR) out.push_back(toUtf8(caps.szPname));
    }
#endif
    return out;
}

}  // namespace ts
