#pragma once
// Minimal WebSocket server (RFC 6455) bound to 127.0.0.1, token-protected.
// Text frames carry JSON-RPC; binary frames carry meter data. Scripts can also send one JSON-RPC
// request per HTTP POST /rpc (token in "Authorization: Bearer <token>" or ?token=).
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ts {

class WsServer {
public:
    struct Client;
    using MessageHandler = std::function<void(const std::shared_ptr<Client>&, const std::string&)>;
    using ConnectHandler = std::function<void(const std::shared_ptr<Client>&)>;

    WsServer();
    ~WsServer();

    /// Binds to 127.0.0.1 on a free port (or `port` if non-zero). Returns the port, 0 on failure.
    int start(const std::string& token, int port = 0);
    void stop();

    void sendText(const std::shared_ptr<Client>& c, const std::string& s);
    void broadcastText(const std::string& s);
    void broadcastBinary(const void* data, size_t len);
    size_t clientCount();

    MessageHandler onMessage;
    ConnectHandler onConnect;
    /// HTTP POST /rpc: request body -> response body (JSON). Runs on the client's thread.
    std::function<std::string(const std::string&)> onHttpRpc;

private:
    void acceptLoop();
    void clientLoop(std::shared_ptr<Client> c);
    bool handshake(Client& c);
    void serveHttpRpc(Client& c, const std::string& req, std::string tok);
    void sendFrame(Client& c, uint8_t opcode, const void* data, size_t len);

    std::string token_;
    intptr_t listen_ = -1;
    std::thread acceptThread_;
    std::atomic<bool> running_{false};
    std::mutex clientsMu_;
    std::vector<std::shared_ptr<Client>> clients_;
};

}  // namespace ts
