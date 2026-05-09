#include "ThreadPool.hpp"


ThreadPool::ThreadPool()
{
    //Should return number of logical cores
    int num_of_threads = std::thread::hardware_concurrency();

    //Work counter is used determine if every added job is executed
    //When new job is added to queue, counter is increased
    //When job is executed, counter is decreased
    //When work counter = 0, we know that every added work has been executed
    work_counter = 0;

    createThreads(num_of_threads);
}

ThreadPool::~ThreadPool()
{
    closeThreads();
}


void ThreadPool::createThreads(int num_of_threads)
{
    exit_flag = false;
    for (int i = 0; i < num_of_threads; i++) {
        threads.push_back(std::thread(&ThreadPool::thread_entry, this));
    }
}

void ThreadPool::closeThreads()
{
    //Set exit flag so that threads know that they are supposed to return
    exit_flag = true;

    //If they are waiting, wake them
    works_cv.notify_all();

    //Join threads
    for (int i = 0; i < threads.size(); i++) {
        threads[i].join();
    }

    threads.clear();
}


void ThreadPool::setThreads(int num_of_threads) {
    wait();

    closeThreads();
    createThreads(num_of_threads);
}



void ThreadPool::wait()
{
    std::unique_lock<std::mutex> lock(works_lock);

    //Wait that work counter becomes 0 which means that all added jobs are executed
    counter_cv.wait(lock, [this] { return work_counter == 0; });
}

void ThreadPool::addWork(const std::function<void()>& job)
{
    {
        //Lock mutex
        std::lock_guard<std::mutex> lock(works_lock);

        //Increase counter and push job to queue
        work_counter += 1;
        works.push(job);
    }

    //Wake one waiting thread
    works_cv.notify_one();
}

void ThreadPool::thread_entry()
{
    bool job_executed = false;
    while (true) {
        std::function<void()> job;

        {
            //Acquire mutex
            std::unique_lock<std::mutex> lock(works_lock);
            if (job_executed) {
                //If job has been executed, decrease counter and notify potentially waiting thread
                work_counter -= 1;
                counter_cv.notify_all();
            }

            //Wait if queue empty
            works_cv.wait(lock, [this] { return !works.empty() || exit_flag; });
            if (exit_flag) {
                break;
            }

            job = works.front();
            works.pop();
        }
        //Mutex is released when scope ends

        //Execute job
        job();

        //Set flag so that we know that we need to decrease work counter
        job_executed = true;
    }
}
