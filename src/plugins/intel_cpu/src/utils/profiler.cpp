
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "profiler.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>



#ifdef __linux__
// performance counter
#    include <linux/perf_event.h>
#    include <sys/mman.h>
#    include <sys/syscall.h>
#    include <unistd.h>

#    define HW_PERF_COUNTER
__attribute__((weak)) int perf_event_open(struct perf_event_attr* attr,
                                          pid_t pid,
                                          int cpu,
                                          int group_fd,
                                          unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

#    include "dumb_vm.hpp"

struct linux_perf_event {
    uint32_t type;
    uint32_t config;
    int fd;
    struct perf_event_mmap_page* buf;
    int uid;

    linux_perf_event(const linux_perf_event&) = delete;
    linux_perf_event(linux_perf_event&&) = delete;

    linux_perf_event(uint32_t type, uint32_t counter, int uid)
        : type(type),
          config(counter),
          fd(-1),
          buf(nullptr),
          uid(uid) {
        struct perf_event_attr attr = {};
        attr.type = type;
        attr.size = PERF_ATTR_SIZE_VER0;
        attr.config = config;
        attr.sample_type = PERF_SAMPLE_READ;
        attr.exclude_kernel = 1;

        fd = perf_event_open(&attr, 0, -1, -1, 0);
        if (fd < 0) {
            perror("perf_event_open");
            return;
        }
        buf = (struct perf_event_mmap_page*)mmap(NULL, sysconf(_SC_PAGESIZE), PROT_READ, MAP_SHARED, fd, 0);
        if (buf == MAP_FAILED) {
            perror("mmap");
            close(fd);
            fd = -1;
            return;
        }
    }
    ~linux_perf_event() {
        if (fd > 0) {
            close(fd);
            munmap(buf, sysconf(_SC_PAGESIZE));
        }
    }
    uint64_t rdpmc_read() {
        uint64_t val, offset;
        uint32_t seq, index;

        do {
            seq = buf->lock;
            std::atomic_thread_fence(std::memory_order_acquire);
            index = buf->index;    //
            offset = buf->offset;  // used to compensate the initial counter value
            if (index == 0) {      /* rdpmc not allowed */
                val = 0;
                std::cout << "rdpmc" << std::endl;
                break;
            }
            val = _rdpmc(index - 1);
            std::atomic_thread_fence(std::memory_order_acquire);
        } while (buf->lock != seq);
        uint64_t ret = (val + offset) & 0xffffffffffff;
        return ret;
    }
};
#endif

namespace ov {
namespace intel_cpu {

// to minimize the overhead of profiler when it's not being enabled,
// the inst is not put inside a singleton function to save extra
// cost in multi-threading safety checks.

struct chromeTrace {
    std::ostream& os;
    int fake_tid;
    uint64_t ts;
    chromeTrace(std::ostream& os, int fake_tid) : os(os), fake_tid(fake_tid) {}
    void setTs(uint64_t _ts) {
        ts = _ts;
    }
    void addCounter(std::string name, std::vector<std::pair<std::string, double>> values) {
        // name += std::to_string(fake_tid);
        os << "{\n"
           << "\"ph\": \"C\",\n"
           << "\"name\": \"" << name << "\",\n"
           << "\"pid\": " << fake_tid << ",\n"
           << "\"tid\": " << 0 << ",\n"
           << "\"ts\": " << ts << ",\n"
           << "\"args\": {\n";
        const char* sep = "";
        for (auto& pair : values) {
            os << sep << "\"" << pair.first << "\" : " << pair.second;
            sep = ",";
        }
        os << " }},\n";
    }
    void addCompleteEvent(std::string name,
                          std::string cat,
                          uint64_t start,
                          uint64_t dur,
                          const std::map<const char*, std::string>& args) {
        os << "{\n";
        os << "\"ph\": \"X\",\n"
           << "\"cat\": \"" << cat << "\",\n"
           << "\"name\": \"" << name << "\",\n"
           << "\"pid\": " << fake_tid << ",\n"
           << "\"tid\": " << 0 << ",\n"
           << "\"ts\": " << start << ",\n"
           << "\"dur\": " << dur << ",\n"
           << "\"args\": {\n";
        const char* sep = "";
        for (auto& a : args) {
            std::string key = a.first;
            if (key.substr(0, 5) == "file:") {
                // special args:
                //     args["file:graph1.dump"] = content
                // will dump the content to a file named `graph1.dump`
                if (!a.second.empty()) {
                    std::ofstream fdump(key.substr(5), std::ios::out);
                    if (fdump.is_open()) {
                        fdump << a.second;
                        fdump.close();
                    }
                }
            } else {
                os << sep << "      \"" << a.first << "\" : \"" << a.second << "\"";
                sep = ",\n";
            }
        }
        os << "\n          }\n";
        os << "},\n";
    }
};

#ifdef HW_PERF_COUNTER

struct PMUMonitor : DumbVM::VM {
    std::map<uint32_t, std::shared_ptr<linux_perf_event>> events;
    bool init_mode;
    int event_uid;

    double rt_duration;
    uint64_t* rt_counters;

    PMUMonitor(const char* config_file_path) {
        std::ifstream input_file(config_file_path);
        if (input_file.is_open()) {
            std::string source((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
            compile(source);
            // disassemble();
            initialize();
        } else {
            init_mode = false;
        }
    }
    void initialize() {
        init_mode = true;
        event_uid = 0;
        execute();
    }
    void call(const std::string& inst, States& s, double& var) override {
        uint32_t perf_type = PERF_TYPE_MAX;

        if (inst == "pmu" || inst == "perf_type_raw")
            perf_type = PERF_TYPE_RAW;
        if (inst == "perf_type_hw_cache")
            perf_type = PERF_TYPE_HW_CACHE;
        if (inst == "perf_type_hardware")
            perf_type = PERF_TYPE_HARDWARE;
        if (inst == "perf_type_software")
            perf_type = PERF_TYPE_SOFTWARE;

        if (perf_type < PERF_TYPE_MAX) {
            uint32_t config = s.stack.back();
            uint32_t evkey = config * PERF_TYPE_MAX + perf_type;
            if (init_mode) {
                // create event in initialization stage
                if (events.count(evkey) == 0) {
                    printf("perf[%d] type=%d, config=0x%x \n", event_uid, perf_type, config);
                    events.emplace(evkey, std::make_shared<linux_perf_event>(perf_type, config, event_uid));
                    event_uid++;
                }
                s.stack.back() = 1;
            } else {
                // get current value
                s.stack.back() = rt_counters[events[evkey]->uid];
            }
            return;
        }

        if (inst == "duration") {
            s.stack.push_back(rt_duration);
            return;
        }
    }

    void record(ProfileCounter& pc) {
        for (auto& ev : events) {
            pc.count[ev.second->uid] = ev.second->rdpmc_read();
        }
    }
    void update(chromeTrace& trace, double dt, uint64_t* counters) {
        rt_duration = dt;
        rt_counters = counters;
        init_mode = false;
        execute();
        for (auto& v : states.vars) {
            if (v.first.substr(0, 6) == "TRACE.")
                trace.addCounter(v.first.substr(6), {{"value", v.second}});
        }
    }
};

void ProfilerManager::addCounter(uint64_t tsc) {
    if (pmum) {
        all_counters.emplace_back(tsc);
        auto& counter = all_counters.back();
        pmum->record(counter);
    }
}

void ProfilerManager::dumpAllCounters(chromeTrace& ct) {
    size_t num_counters = sizeof(all_counters[0].count) / sizeof(all_counters[0].count[0]);
    std::vector<uint64_t> dc(num_counters, 0);

    int i0 = 0;
    for (int i = 1; i < all_counters.size(); i++) {
        auto& pc0 = all_counters[i0];
        auto& pc = all_counters[i];

        // only compute metrics for events 10us apart to avoid outliers
        double dt = (pc.end - pc0.end) * 1.0 / tsc_ticks_per_second;
        if (dt < 10e-6)
            continue;

        for (int k = 0; k < dc.size(); k++) {
            dc[k] = pc.count[k] - pc0.count[k];
        }
        ct.setTs(tsc_to_usec(pc0.end));
        if (pmum) pmum->update(ct, dt, &dc[0]);

        i0 = i;
    }
}
#else
struct PMUMonitor {
    PMUMonitor(const char* config_file_path) {}
};
ProfileCounter & ProfilerManager::addCounter() {
    all_counters.push_back({});
    auto& counter = all_counters.back();
    return counter;
}
void ProfilerManager::dumpAllCounters(chromeTrace& ct) {}
#endif

thread_local ProfilerManager profilerManagerInstance;

static std::atomic_int totalProfilerManagers{0};

bool ProfilerInit() {
    return profilerManagerInstance.is_enabled();
}

static uint64_t rdtsc_calibrate(int seconds = 1) {
    uint64_t start_ticks;
    start_ticks = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    return (__rdtsc() - start_ticks) / seconds;
}

ProfileData::ProfileData() {
    start = __rdtsc();
    profilerManagerInstance.addCounter(start);
}

ProfileData::ProfileData(const std::string& name) : name(name) {
    start = __rdtsc();
    profilerManagerInstance.addCounter(start);
}

void ProfileData::record_end(ProfileData* p) {
    p->end = __rdtsc();
    profilerManagerInstance.addCounter(p->end);
}

std::atomic<uint64_t> tsc_ticks_per_second(0);
std::atomic<uint64_t> tsc_ticks_base(0);
bool profile_enabled = false;
std::shared_ptr<ProfileData> profile_data_null;

ProfilerManager::ProfilerManager() {
    const char* str_enable = std::getenv("OV_CPU_PROFILE");
    if (!str_enable)
        str_enable = "0";
    int num_hint = atoi(str_enable);
    set_enable(num_hint > 0);
    if (enabled) {
        pmu = std::make_shared<PMUMonitor>("ov_pmu.txt");
        pmum = reinterpret_cast<PMUMonitor*>(pmu.get());
        if (!pmum->init_mode) {
            pmum = nullptr;
        }

        if (tsc_ticks_per_second == 0) {
            uint64_t expected = 0;
            auto tps = rdtsc_calibrate();
            tsc_ticks_per_second.compare_exchange_strong(expected, tps);
            std::cout << "=== ProfilerManager: tsc_ticks_per_second = " << tsc_ticks_per_second << std::endl;
            tsc_ticks_base.compare_exchange_strong(expected, __rdtsc());
        }
    }

    std::cout << "=== ProfilerManager: is " << (enabled? "enabled":"disabled") << " ====" << std::endl;

    tid = std::this_thread::get_id();
    totalProfilerManagers++;
}

static std::mutex dump_mutex;
static std::string dump_text_all;
static std::atomic<int> total_traces(0);
static std::atomic<int> fake_tids;

void ProfilerManager::finalize() {
    // collect all entries
    if (all_data.size()) {
        std::lock_guard<std::mutex> lock_g(dump_mutex);
        auto fake_tid = fake_tids.fetch_add(1);
        std::stringstream dump_ss;
        chromeTrace ct(dump_ss, fake_tid);
        for (auto& d : all_data) {
            ct.addCompleteEvent(d.name, d.cat, tsc_to_usec(d.start), tsc_to_usec(d.end) - tsc_to_usec(d.start), d.args);
            total_traces++;
        }
        dumpAllCounters(ct);
        dump_text_all += dump_ss.str();
        std::cout << "==== Profile: total number of profile entries " << all_data.size() << "," << all_counters.size()
                  << std::endl;
    }

    if (totalProfilerManagers.fetch_sub(1) != 1)
        return;

    if (total_traces == 0)
        return;

    // the last ProfilerManagers is responsible for dump to file
    const char* dump_file_name = "ov_profile.json";
    std::ofstream fw(dump_file_name, std::ios::out);

    if (fw.is_open()) {
        fw << "{\n";
        fw << "\"schemaVersion\": 1,\n";
        fw << "\"traceEvents\": [\n";

        fw << dump_text_all;

        fw << R"({
            "name": "Profiler End",
            "ph": "i",
            "s": "g",
            "pid": "Traces",
            "tid": "Trace OV Profiler",
            "ts":)"
           << tsc_to_usec(__rdtsc()) << "}",
            fw << "]\n";
        fw << "}\n";
        fw.close();
        std::cout << "==== Profile data is dumpped into " << dump_file_name << "\n";
    }
}

ProfilerManager::~ProfilerManager() {
    finalize();
}

void ProfilerManager::set_enable(bool on) {
    if (enabled != on) {
        enabled = on;
    }
    profile_enabled = on;
}

}  // namespace intel_cpu
}  // namespace ov