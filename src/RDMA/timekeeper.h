#pragma once

#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

class TimeKeeper {
public:
  std::map<std::string, unsigned long> times{};
  std::mutex mtx;

  void snapshot(std::string name, bool overwrite = false) {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now.time_since_epoch())
                        .count();

    std::unique_lock<std::mutex> lock(mtx);

    auto it = times.find(name);
    if (it != times.end()) {
      if (overwrite && duration > it->second) {
        it->second = duration;
      } else {
        lock.unlock();
        return;
      }
    } else {
      times[name] = duration;
    }
    lock.unlock();
  }

  void add_time(std::string name, unsigned long value) {
    std::unique_lock<std::mutex> lock(mtx);
    times[name] = value;
    lock.unlock();
  }

  unsigned long get_duration(std::string first, std::string second) {
    return times[first] - times[second];
  }

  void print_history() {
    for (const auto &[name, time] : times) {
      std::cout << name << ": " << time << " ns" << std::endl;
    }
  }
};
