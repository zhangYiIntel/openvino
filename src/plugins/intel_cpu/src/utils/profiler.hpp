// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0Add commentMore actions
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
#include <functional>

extern "C" {
#ifdef _WIN32
#    include <intrin.h>
#else
#    include <x86intrin.h>
#endif
}

namespace ov {
namespace intel_cpu {

using ProfileArgs = std::map<const char *, std::string>;

struct ProfileData {
    uint64_t start;
    uint64_t end;
    std::string name;  // Title
    std::string cat;   // Category
    ProfileArgs args;

    ProfileData();
    ProfileData(const std::string& name);
    static void record_end(ProfileData* p);
};

struct ProfileCounter {
    uint64_t end;
    uint64_t count[4];
    ProfileCounter(uint64_t end) : end(end) {}
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

    ProfileData* startProfile() {
        all_data.emplace_back();
        return &all_data.back();
    }
    uint64_t tsc_to_usec(uint64_t tsc_ticks) {
        return (tsc_ticks - tsc_ticks_base) * 1000000 / tsc_ticks_per_second;
    }

    void addCounter(uint64_t tsc);

    void set_enable(bool on);
    bool is_enabled() {
        return enabled;
    }

    void dumpAllCounters(chromeTrace& ct);

    friend class Profiler;
};

extern thread_local ProfilerManager profilerManagerInstance;

extern bool profile_enabled;
extern std::shared_ptr<ProfileData> profile_data_null;

bool ProfilerInit();

#define ENABLE_CPU_PROFILER 1
#if ENABLE_CPU_PROFILER == 1
inline std::shared_ptr<ProfileData> Profile(const char * name) {
    if (!profile_enabled)
        return profile_data_null;
    ProfileData* p = profilerManagerInstance.startProfile();
    p->name = name;
    return std::shared_ptr<ProfileData>(p, ProfileData::record_end);
}

// getArgs return ProfileArgs
template<typename F>
inline std::shared_ptr<ProfileData> Profile(F fill_data) {
    static_assert(std::is_convertible<F, std::function<void(ProfileData *)>>::value, "Wrong Signature!");

    if (!profile_enabled)
        return profile_data_null;

    ProfileData* p = profilerManagerInstance.startProfile();
    fill_data(p);
    return std::shared_ptr<ProfileData>(p, ProfileData::record_end);
}
#else
#define Profile(...) 1
#endif

}  // namespace intel_cpu
}  // namespace ov