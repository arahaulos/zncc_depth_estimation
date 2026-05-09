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
    //ThreadPool has queue of works
    //Works are lambdas wrapped in std::function
    //ThreadPool has N threads, which will fetch works from queue

    ThreadPool();
    ~ThreadPool();

    void setThreads(int threads);

    void thread_entry();

    void wait(); //Blocks until all queued works are executed
    void addWork(const std::function<void()>& job);
private:
    void createThreads(int num_of_threads);
    void closeThreads();

    std::queue<std::function<void()>> works;
    std::vector<std::thread> threads;
    std::atomic<bool> exit_flag;

    std::mutex works_lock;
    std::condition_variable works_cv;

    std::condition_variable counter_cv;

    uint64_t work_counter;
};
