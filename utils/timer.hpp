# pragma once

# include <chrono>
# include <iostream>

using namespace std;

class Timer {
private:
    bool in_tick = false;
    chrono::time_point <chrono::steady_clock> start;
    chrono::time_point <chrono::steady_clock> finish;
    long long total_time;
    string name;
public:
    Timer(string name);

    void tick();

    void untick();

    long double get_total_time();

    void log();
};