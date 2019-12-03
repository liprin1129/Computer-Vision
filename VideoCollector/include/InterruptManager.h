#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>

class InterruptManager {
    public:
        void keyInputInterrupt(std::mutex &threadLockMutex, char &key) {
            while(key!='q') {
                //threadLockMutex.lock();
                //std::cout << "Input: \n";
                std::cin >> key;
                //key = std::getchar();
                std::this_thread::sleep_for(std::chrono::seconds(1));
                //threadLockMutex.unlock();
            }
        };
};