#pragma once

#include <functional>
#include <vector>
#include <queue>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

struct ThreadPool
{
    ThreadPool();
    ~ThreadPool();

    void thread_entry();

    void wait();
    void addWork(const std::function<void()>& job);
private:
    std::queue<std::function<void()>> works;
    std::vector<std::thread> threads;
    std::atomic<bool> exit_flag;

    std::mutex works_lock;
    std::condition_variable works_cv;

    std::condition_variable counter_cv;

    uint64_t work_counter;
};
