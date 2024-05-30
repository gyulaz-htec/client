// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <curl/curl.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

// The options for authorizing and authenticating SSL/TLS connections.
struct HttpSslOptions {
  enum CERTTYPE { CERT_PEM = 0, CERT_DER = 1 };
  enum KEYTYPE { KEY_PEM = 0, KEY_DER = 1 };
  explicit HttpSslOptions()
      : verify_peer(1), verify_host(2), cert_type(CERTTYPE::CERT_PEM),
        key_type(KEYTYPE::KEY_PEM)
  {
  }
  long verify_peer;
  long verify_host;
  std::string ca_info;
  CERTTYPE cert_type;
  std::string cert;
  KEYTYPE key_type;
  std::string key;
};

class HttpRequest {
 public:
  HttpRequest(
      std::function<void(HttpRequest*)>&& completion_callback,
      const bool verbose = false);
  virtual ~HttpRequest();
  void AddInput(uint8_t* buf, size_t byte_size);
  void GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes);
  std::string response_buffer_;
  size_t total_input_byte_size_{0};
  uint32_t http_code_{200};
  std::function<void(HttpRequest*)> completion_callback_{nullptr};
  struct curl_slist* header_list_{nullptr};

 protected:
  const bool verbose_{false};
  std::deque<std::pair<uint8_t*, size_t>> data_buffers_;
};

class HttpClient {
 public:
  enum class CompressionType { NONE, DEFLATE, GZIP };
  virtual ~HttpClient();

 protected:
  void SetSSLCurlOptions(CURL* curl_handle);
  HttpClient(
      const std::string& server_url, bool verbose = false,
      const HttpSslOptions& ssl_options = HttpSslOptions());
  void Send(CURL* handle, std::unique_ptr<HttpRequest>&& request);

 protected:
  void AsyncTransfer();

  std::atomic<bool> exiting_{false};
  std::thread worker_;
  std::condition_variable cv_;
  std::atomic<bool> has_new_request_{false};

  const std::string url_;
  HttpSslOptions ssl_options_;
  void* multi_handle_;
  std::queue<std::unique_ptr<HttpRequest>> request_queue_;
  using AsyncReqMap = std::map<uintptr_t, std::unique_ptr<HttpRequest>>;
  AsyncReqMap ongoing_async_requests_;
  std::mutex mutex_;
  bool verbose_;

 private:
  const std::string& ParseSslKeyType(HttpSslOptions::KEYTYPE key_type);
  const std::string& ParseSslCertType(HttpSslOptions::CERTTYPE cert_type);
  static std::mutex curl_init_mtx_;
};

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
