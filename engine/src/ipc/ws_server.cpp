#include "ipc/ws_server.h"

#include <algorithm>
#include <cstring>
#include <sstream>

#include "util/encoding.h"
#include "util/log.h"
#include "util/platform.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using SockLen = int;
static void closeSock(intptr_t s) { closesocket(static_cast<SOCKET>(s)); }
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using SockLen = socklen_t;
// shutdown() first: on Linux, close() alone does not wake a thread blocked in accept()/recv().
static void closeSock(intptr_t s) {
    ::shutdown(static_cast<int>(s), SHUT_RDWR);
    ::close(static_cast<int>(s));
}
#endif

namespace nt {

struct WsServer::Client {
    intptr_t sock = -1;
    std::mutex sendMu;
    std::thread thread;
    std::atomic<bool> alive{true};
    std::atomic<bool> ready{false};  // handshake done: may receive frames
    std::string pending;  // bytes read beyond the HTTP handshake
};

namespace {

bool recvAll(intptr_t s, void* buf, size_t len) {
    auto* p = static_cast<char*>(buf);
    while (len > 0) {
        const int n = recv(static_cast<decltype(socket(0, 0, 0))>(s), p, static_cast<int>(std::min<size_t>(len, 1 << 20)), 0);
        if (n <= 0) return false;
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

bool sendAll(intptr_t s, const void* buf, size_t len) {
    const auto* p = static_cast<const char*>(buf);
    while (len > 0) {
        const int n = send(static_cast<decltype(socket(0, 0, 0))>(s), p, static_cast<int>(std::min<size_t>(len, 1 << 20)), 0);
        if (n <= 0) return false;
        p += n;
        len -= static_cast<size_t>(n);
    }
    return true;
}

std::string headerValue(const std::string& req, const std::string& name) {
    std::istringstream in(req);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (key == name) {
            std::string v = line.substr(colon + 1);
            v.erase(0, v.find_first_not_of(" \t"));
            return v;
        }
    }
    return {};
}

}  // namespace

WsServer::WsServer() {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

WsServer::~WsServer() {
    stop();
#ifdef _WIN32
    WSACleanup();
#endif
}

int WsServer::start(const std::string& token, int port) {
    token_ = token;
    const auto s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == static_cast<decltype(s)>(-1)) return 0;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(s, 8) != 0) {
        closeSock(static_cast<intptr_t>(s));
        return 0;
    }
    SockLen len = sizeof(addr);
    getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len);
    listen_ = static_cast<intptr_t>(s);
    running_ = true;
    acceptThread_ = std::thread([this] { acceptLoop(); });
    return ntohs(addr.sin_port);
}

void WsServer::stop() {
    if (!running_.exchange(false)) return;
    closeSock(listen_);
    if (acceptThread_.joinable()) acceptThread_.join();
    std::vector<std::shared_ptr<Client>> cs;
    {
        std::lock_guard lock(clientsMu_);
        cs.swap(clients_);
    }
    for (auto& c : cs) {
        c->alive = false;
        closeSock(c->sock);
        if (c->thread.joinable()) c->thread.join();
    }
}

size_t WsServer::clientCount() {
    std::lock_guard lock(clientsMu_);
    return static_cast<size_t>(std::count_if(clients_.begin(), clients_.end(), [](auto& c) { return c->ready.load(); }));
}

void WsServer::acceptLoop() {
    setThreadName("nt-ipc-accept");
    while (running_) {
        sockaddr_in peer{};
        SockLen len = sizeof(peer);
        const auto s = accept(static_cast<decltype(socket(0, 0, 0))>(listen_), reinterpret_cast<sockaddr*>(&peer), &len);
        if (s == static_cast<decltype(s)>(-1)) {
            if (!running_) break;
            continue;
        }
        if (peer.sin_addr.s_addr != htonl(INADDR_LOOPBACK)) {
            closeSock(static_cast<intptr_t>(s));
            continue;
        }
        int one = 1;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
        auto c = std::make_shared<Client>();
        c->sock = static_cast<intptr_t>(s);
        // Hold the lock while starting the thread so its cleanup can't race the assignment.
        std::lock_guard lock(clientsMu_);
        clients_.push_back(c);
        c->thread = std::thread([this, c] { clientLoop(c); });
    }
}

bool WsServer::handshake(Client& c) {
    std::string req;
    char buf[1024];
    while (req.find("\r\n\r\n") == std::string::npos) {
        const int n = recv(static_cast<decltype(socket(0, 0, 0))>(c.sock), buf, sizeof(buf), 0);
        if (n <= 0 || req.size() > 16384) return false;
        req.append(buf, static_cast<size_t>(n));
    }
    const size_t end = req.find("\r\n\r\n") + 4;
    c.pending = req.substr(end);
    req.resize(end);
    // Request line: GET /?token=... HTTP/1.1
    const size_t sp1 = req.find(' '), sp2 = req.find(' ', sp1 + 1);
    const std::string target = sp1 == std::string::npos ? "" : req.substr(sp1 + 1, sp2 - sp1 - 1);
    const size_t tpos = target.find("token=");
    std::string tok = tpos == std::string::npos ? "" : target.substr(tpos + 6);
    if (const size_t amp = tok.find('&'); amp != std::string::npos) tok.resize(amp);
    const std::string key = headerValue(req, "sec-websocket-key");
    const std::string origin = headerValue(req, "origin");
    if (tok != token_ || key.empty()) {
        const char* deny = "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        sendAll(c.sock, deny, std::strlen(deny));
        return false;
    }
    const auto digest = sha1(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    const std::string accept = base64Encode(digest.data(), digest.size());
    const std::string resp = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: " + accept + "\r\n\r\n";
    return sendAll(c.sock, resp.data(), resp.size());
}

void WsServer::clientLoop(std::shared_ptr<Client> c) {
    setThreadName("nt-ipc-client");
    if (handshake(*c)) {
        c->ready = true;
        if (onConnect) onConnect(c);
        std::string message;
        uint8_t messageOp = 0;
        auto readBytes = [&](void* dst, size_t n) -> bool {
            // Serve leftover handshake bytes first.
            auto* d = static_cast<char*>(dst);
            const size_t take = std::min(n, c->pending.size());
            if (take) {
                std::memcpy(d, c->pending.data(), take);
                c->pending.erase(0, take);
            }
            return take == n || recvAll(c->sock, d + take, n - take);
        };
        while (c->alive && running_) {
            uint8_t h[2];
            if (!readBytes(h, 2)) break;
            const bool fin = h[0] & 0x80;
            const uint8_t op = h[0] & 0x0f;
            const bool masked = h[1] & 0x80;
            uint64_t len = h[1] & 0x7f;
            if (len == 126) {
                uint8_t e[2];
                if (!readBytes(e, 2)) break;
                len = (static_cast<uint64_t>(e[0]) << 8) | e[1];
            } else if (len == 127) {
                uint8_t e[8];
                if (!readBytes(e, 8)) break;
                len = 0;
                for (int i = 0; i < 8; ++i) len = (len << 8) | e[i];
            }
            if (len > (16u << 20)) break;  // 16 MB cap
            uint8_t mask[4] = {0, 0, 0, 0};
            if (masked && !readBytes(mask, 4)) break;
            std::string payload(static_cast<size_t>(len), '\0');
            if (len && !readBytes(payload.data(), static_cast<size_t>(len))) break;
            if (masked)
                for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<char>(payload[i] ^ mask[i & 3]);
            if (op == 0x8) {  // close
                sendFrame(*c, 0x8, nullptr, 0);
                break;
            }
            if (op == 0x9) {  // ping
                sendFrame(*c, 0xA, payload.data(), payload.size());
                continue;
            }
            if (op == 0xA) continue;
            if (op == 0x1 || op == 0x2) {
                message = std::move(payload);
                messageOp = op;
            } else if (op == 0x0) {
                message += payload;
            }
            if (fin && messageOp == 0x1 && onMessage) onMessage(c, message);
        }
    }
    c->alive = false;
    closeSock(c->sock);
    std::lock_guard lock(clientsMu_);
    auto it = std::find(clients_.begin(), clients_.end(), c);
    if (it != clients_.end()) {  // otherwise stop() owns (and joins) this thread
        clients_.erase(it);
        c->thread.detach();
    }
}

void WsServer::sendFrame(Client& c, uint8_t opcode, const void* data, size_t len) {
    uint8_t h[10];
    size_t hl = 2;
    h[0] = static_cast<uint8_t>(0x80 | opcode);
    if (len < 126) {
        h[1] = static_cast<uint8_t>(len);
    } else if (len < 65536) {
        h[1] = 126;
        h[2] = static_cast<uint8_t>(len >> 8);
        h[3] = static_cast<uint8_t>(len);
        hl = 4;
    } else {
        h[1] = 127;
        for (int i = 0; i < 8; ++i) h[2 + i] = static_cast<uint8_t>(static_cast<uint64_t>(len) >> (56 - 8 * i));
        hl = 10;
    }
    std::lock_guard lock(c.sendMu);
    if (!c.alive || !c.ready) return;
    if (!sendAll(c.sock, h, hl) || (len && !sendAll(c.sock, data, len))) c.alive = false;
}

void WsServer::sendText(const std::shared_ptr<Client>& c, const std::string& s) { sendFrame(*c, 0x1, s.data(), s.size()); }

void WsServer::broadcastText(const std::string& s) {
    std::vector<std::shared_ptr<Client>> cs;
    {
        std::lock_guard lock(clientsMu_);
        cs = clients_;
    }
    for (auto& c : cs) sendFrame(*c, 0x1, s.data(), s.size());
}

void WsServer::broadcastBinary(const void* data, size_t len) {
    std::vector<std::shared_ptr<Client>> cs;
    {
        std::lock_guard lock(clientsMu_);
        cs = clients_;
    }
    for (auto& c : cs) sendFrame(*c, 0x2, data, len);
}

}  // namespace nt
