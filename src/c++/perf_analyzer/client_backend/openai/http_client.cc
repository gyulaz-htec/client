#include "http_client.h"

#include <functional>
#include <iostream>

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace openai {

HttpRequest::HttpRequest(
    std::function<void(HttpRequest*)>&& completion_callback, const bool verbose)
    : completion_callback_(std::move(completion_callback)), verbose_(verbose)
{
  std::cout << "HttpRequest created" << std::endl;
}

HttpRequest::~HttpRequest()
{
  if (header_list_ != nullptr) {
    curl_slist_free_all(header_list_);
    header_list_ = nullptr;
  }
  std::cout << "HttpRequest destroyed" << std::endl;
}

void
HttpRequest::AddInput(uint8_t* buf, size_t byte_size)
{
  data_buffers_.push_back(std::pair<uint8_t*, size_t>(buf, byte_size));
  total_input_byte_size_ += byte_size;
}

void
HttpRequest::GetNextInput(uint8_t* buf, size_t size, size_t* input_bytes)
{
  *input_bytes = 0;

  while (!data_buffers_.empty() && size > 0) {
    const size_t csz = std::min(data_buffers_.front().second, size);
    if (csz > 0) {
      const uint8_t* input_ptr = data_buffers_.front().first;
      std::copy(input_ptr, input_ptr + csz, buf);
      size -= csz;
      buf += csz;
      *input_bytes += csz;

      data_buffers_.front().first += csz;
      data_buffers_.front().second -= csz;
    }
    if (data_buffers_.front().second == 0) {
      data_buffers_.pop_front();
    }
  }
}

std::mutex HttpClient::curl_init_mtx_{};
HttpClient::HttpClient(
    const std::string& server_url, bool verbose,
    const HttpSslOptions& ssl_options)
    : url_(server_url), verbose_(verbose), ssl_options_(ssl_options)
{
  {
    std::lock_guard<std::mutex> lk(curl_init_mtx_);
    if (curl_global_init(CURL_GLOBAL_ALL) != 0) {
      throw std::runtime_error("CURL global initialization failed");
    }
  }

  multi_handle_ = curl_multi_init();
  worker_ = std::thread(&HttpClient::AsyncTransfer, this);
  std::cout << "HttpClient created" << std::endl;
}

HttpClient::~HttpClient()
{
  exiting_ = true;
  std::cout << "Exiting HttpClient" << std::endl;
  cv_.notify_all();
  std::cout << "Notified all" << std::endl;
  if (worker_.joinable()) {
    std::cout << "Joining worker thread" << std::endl;
    worker_.join();
  }
  std::cout << "Worker thread joined" << std::endl;

  while (!request_queue_.empty()) {
    request_queue_.pop();
    std::cout << "Request queue popped" << std::endl;
  }

  std::cout << "Request queue cleared" << std::endl;
  curl_multi_cleanup(multi_handle_);
  std::cout << "Multi handle cleaned up" << std::endl;
  {
    std::lock_guard<std::mutex> lk(curl_init_mtx_);
    std::cout << "Lock guard acquired" << std::endl;
    curl_global_cleanup();
    std::cout << "CURL global cleanup" << std::endl;
  }
  std::cout << "HttpClient destroyed" << std::endl;
}

const std::string&
HttpClient::ParseSslCertType(HttpSslOptions::CERTTYPE cert_type)
{
  static std::string pem_str{"PEM"};
  static std::string der_str{"DER"};
  switch (cert_type) {
    case HttpSslOptions::CERTTYPE::CERT_PEM:
      return pem_str;
    case HttpSslOptions::CERTTYPE::CERT_DER:
      return der_str;
  }
  throw std::runtime_error(
      "Unexpected SSL certificate type encountered. Only PEM and DER are "
      "supported.");
}

const std::string&
HttpClient::ParseSslKeyType(HttpSslOptions::KEYTYPE key_type)
{
  static std::string pem_str{"PEM"};
  static std::string der_str{"DER"};
  switch (key_type) {
    case HttpSslOptions::KEYTYPE::KEY_PEM:
      return pem_str;
    case HttpSslOptions::KEYTYPE::KEY_DER:
      return der_str;
  }
  throw std::runtime_error(
      "Unsupported SSL key type encountered. Only PEM and DER are supported.");
}

void
HttpClient::SetSSLCurlOptions(CURL* curl_handle)
{
  curl_easy_setopt(
      curl_handle, CURLOPT_SSL_VERIFYPEER, ssl_options_.verify_peer);
  curl_easy_setopt(
      curl_handle, CURLOPT_SSL_VERIFYHOST, ssl_options_.verify_host);
  if (!ssl_options_.ca_info.empty()) {
    curl_easy_setopt(curl_handle, CURLOPT_CAINFO, ssl_options_.ca_info.c_str());
  }
  const auto& curl_cert_type = ParseSslCertType(ssl_options_.cert_type);
  curl_easy_setopt(curl_handle, CURLOPT_SSLCERTTYPE, curl_cert_type.c_str());
  if (!ssl_options_.cert.empty()) {
    curl_easy_setopt(curl_handle, CURLOPT_SSLCERT, ssl_options_.cert.c_str());
  }
  const auto& curl_key_type = ParseSslKeyType(ssl_options_.key_type);
  curl_easy_setopt(curl_handle, CURLOPT_SSLKEYTYPE, curl_key_type.c_str());
  if (!ssl_options_.key.empty()) {
    curl_easy_setopt(curl_handle, CURLOPT_SSLKEY, ssl_options_.key.c_str());
  }
}

void
HttpClient::Send(CURL* handle, std::unique_ptr<HttpRequest>&& request)
{
  std::cout << "Send called" << std::endl;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ongoing_async_requests_.emplace(
        reinterpret_cast<uintptr_t>(handle), std::move(request));
    curl_multi_add_handle(multi_handle_, handle);
    std::cout << "Request added to ongoing_async_requests_" << std::endl;
  }
  has_new_request_ = true;
  cv_.notify_all();
}

void
HttpClient::AsyncTransfer()
{
  int place_holder = 0;
  CURLMsg* msg = nullptr;

  while (true) {
    std::vector<std::unique_ptr<HttpRequest>> request_list;

    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return has_new_request_ || exiting_; });
      if (exiting_ && ongoing_async_requests_.empty()) {
        std::cout << "Exiting AsyncTransfer loop" << std::endl;
        break;
      }
      has_new_request_ = false;
      std::cout << "New work detected, continuing AsyncTransfer" << std::endl;
    }

    CURLMcode mc = curl_multi_perform(multi_handle_, &place_holder);
    if (mc != CURLM_OK) {
      std::cerr << "Unexpected error: curl_multi_perform failed. Code:" << mc
                << std::endl;
      continue;
    }

    int numfds;
    mc = curl_multi_poll(multi_handle_, NULL, 0, 1000, &numfds);
    if (mc != CURLM_OK) {
      std::cerr << "Unexpected error: curl_multi_poll failed. Code:" << mc
                << std::endl;
      continue;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      while ((msg = curl_multi_info_read(multi_handle_, &place_holder))) {
        uintptr_t identifier = reinterpret_cast<uintptr_t>(msg->easy_handle);
        auto itr = ongoing_async_requests_.find(identifier);
        if (itr == ongoing_async_requests_.end()) {
          std::cerr
              << "Unexpected error: received completed request that is not "
              << "in the list of asynchronous requests" << std::endl;
          curl_multi_remove_handle(multi_handle_, msg->easy_handle);
          curl_easy_cleanup(msg->easy_handle);
          continue;
        }

        uint32_t http_code = 400;
        if (msg->data.result == CURLE_OK) {
          curl_easy_getinfo(
              msg->easy_handle, CURLINFO_RESPONSE_CODE, &http_code);
        } else if (msg->data.result == CURLE_OPERATION_TIMEDOUT) {
          http_code = 499;
        }

        request_list.emplace_back(std::move(itr->second));
        ongoing_async_requests_.erase(itr);
        curl_multi_remove_handle(multi_handle_, msg->easy_handle);
        curl_easy_cleanup(msg->easy_handle);

        std::unique_ptr<HttpRequest>& async_request = request_list.back();
        async_request->http_code_ = http_code;

        if (msg->msg != CURLMSG_DONE) {
          std::cerr << "Unexpected error: received CURLMsg=" << msg->msg
                    << std::endl;
        }
      }
    }

    for (auto& this_request : request_list) {
      this_request->completion_callback_(this_request.get());
      std::cout << "Completed request processed" << std::endl;
    }
  }
  std::cout << "AsyncTransfer exiting" << std::endl;
}

}}}}  // namespace triton::perfanalyzer::clientbackend::openai
