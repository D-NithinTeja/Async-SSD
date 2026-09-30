// thread_safe_queue.h
// Blocking FIFO queue for communication between benchmark worker threads.

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

template <typename T>
class ThreadSafeQueue {
 public:
  void push(T value) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(std::move(value));
    }
    not_empty_.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] { return !queue_.empty(); });
    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

 private:
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::queue<T> queue_;
};
