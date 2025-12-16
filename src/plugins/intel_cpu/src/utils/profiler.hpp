// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace ov {
namespace intel_cpu {

struct ProfileData {
    uint64_t start;
    uint64_t end;
    std::string name;  // Title
    std::string cat;   // Category
    std::map<const char *, std::string> args;

    ProfileData(const std::string& name);
    static void record_end(ProfileData* p);
};

struct ProfileCounter {
    uint64_t end;
    uint64_t count[4];
    ProfileCounter() = default;
};

class Profiler;
class chromeTrace;
struct PMUMonitor;

extern std::atomic<uint64_t>  tsc_ticks_per_second;
extern std::atomic<uint64_t>  tsc_ticks_base;

class ProfilerManager {
    bool enabled;
    // cannot use vector<> since each new Profile() API call will
    // emplace_back() an item and wrap it into a shared_ptr, this
    // process is nested and during which vector resize may invalid
    // the ProfileData elements still referenced by an alive shared_ptr
    // and later when it finally gets un-referenced, a wild pointer would
    // be updated and memory would be corrupted. deque can fix it.
    std::deque<ProfileData> all_data;
    std::thread::id tid;
    std::deque<ProfileCounter> all_counters;

    std::shared_ptr<void> pmu;
    PMUMonitor * pmum;

public:
    ProfilerManager();
    ~ProfilerManager();

    void finalize();

    ProfileData* startProfile(const std::string& name) {
        all_data.emplace_back(name);
        return &all_data.back();
    }
    uint64_t tsc_to_usec(uint64_t tsc_ticks) {
        return (tsc_ticks - tsc_ticks_base) * 1000000 / tsc_ticks_per_second;
    }

    ProfileCounter & addCounter();

    void set_enable(bool on);
    bool is_enabled() {
        return enabled;
    }

    void dumpAllCounters(chromeTrace& ct);

    friend class Profiler;
};

extern thread_local ProfilerManager profilerManagerInstance;

inline std::shared_ptr<ProfileData> Profile(const char* name) {
    if (!profilerManagerInstance.is_enabled())
        return nullptr;
    ProfileData* p = profilerManagerInstance.startProfile(name);
    return std::shared_ptr<ProfileData>(p, ProfileData::record_end);
}

inline std::shared_ptr<ProfileData> Profile(const std::string& name) {
    if (!profilerManagerInstance.is_enabled())
        return nullptr;
    ProfileData* p = profilerManagerInstance.startProfile(name);
    return std::shared_ptr<ProfileData>(p, ProfileData::record_end);
}

}  // namespace intel_cpu
}  // namespace ov